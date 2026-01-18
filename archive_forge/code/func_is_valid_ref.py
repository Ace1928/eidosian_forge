from __future__ import annotations
import re
import typing as t
from .util import (
def is_valid_ref(self, ref: str) -> bool:
    """Return True if the given reference is valid, otherwise return False."""
    cmd = ['show', ref]
    try:
        self.run_git(cmd, str_errors='replace')
        return True
    except SubprocessError:
        return False