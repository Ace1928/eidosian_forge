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
def requires_to_requires_dist(requirement: Requirement) -> str:
    """Return the version specifier for a requirement in PEP 345/566 fashion."""
    if getattr(requirement, 'url', None):
        return ' @ ' + requirement.url
    requires_dist = []
    for spec in requirement.specifier:
        requires_dist.append(spec.operator + spec.version)
    if requires_dist:
        return ' ' + ','.join(sorted(requires_dist))
    else:
        return ''