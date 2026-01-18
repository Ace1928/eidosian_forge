from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def create_result_name(self, extension: str) -> str:
    """Return the name of the result file using the given extension."""
    name = 'ansible-test-%s' % self.command
    if self.test:
        name += '-%s' % self.test
    if self.python_version:
        name += '-python-%s' % self.python_version
    name += extension
    return name