from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
@cache
def remote_completion() -> dict[str, PosixRemoteCompletionConfig]:
    """Return remote completion entries."""
    return load_completion('remote', PosixRemoteCompletionConfig)