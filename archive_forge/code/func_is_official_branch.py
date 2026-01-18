from __future__ import annotations
import os
import platform
import random
import re
import typing as t
from ..config import (
from ..io import (
from ..git import (
from ..util import (
from . import (
def is_official_branch(self, name: str) -> bool:
    """Return True if the given branch name an official branch for development or releases."""
    if self.args.base_branch:
        return name == self.args.base_branch
    if name == 'devel':
        return True
    if re.match('^stable-[0-9]+\\.[0-9]+$', name):
        return True
    return False