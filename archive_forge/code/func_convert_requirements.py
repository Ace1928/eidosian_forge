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
def convert_requirements(requirements: list[str]) -> Iterator[str]:
    """Yield Requires-Dist: strings for parsed requirements strings."""
    for req in requirements:
        parsed_requirement = Requirement(req)
        spec = requires_to_requires_dist(parsed_requirement)
        extras = ','.join(sorted((safe_extra(e) for e in parsed_requirement.extras)))
        if extras:
            extras = f'[{extras}]'
        yield (safe_name(parsed_requirement.name) + extras + spec)