from __future__ import annotations
import functools
import itertools
import os.path
import re
import textwrap
from email.message import Message
from email.parser import Parser
from typing import Iterator
from .vendored.packaging.requirements import Requirement
def generate_requirements(extras_require: dict[str, list[str]]) -> Iterator[tuple[str, str]]:
    """
    Convert requirements from a setup()-style dictionary to
    ('Requires-Dist', 'requirement') and ('Provides-Extra', 'extra') tuples.

    extras_require is a dictionary of {extra: [requirements]} as passed to setup(),
    using the empty extra {'': [requirements]} to hold install_requires.
    """
    for extra, depends in extras_require.items():
        condition = ''
        extra = extra or ''
        if ':' in extra:
            extra, condition = extra.split(':', 1)
        extra = safe_extra(extra)
        if extra:
            yield ('Provides-Extra', extra)
            if condition:
                condition = '(' + condition + ') and '
            condition += "extra == '%s'" % extra
        if condition:
            condition = ' ; ' + condition
        for new_req in convert_requirements(depends):
            yield ('Requires-Dist', new_req + condition)