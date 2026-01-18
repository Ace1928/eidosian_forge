from __future__ import annotations
import asyncio
import gc
import os
import socket as stdlib_socket
import sys
import warnings
from contextlib import closing, contextmanager
from typing import TYPE_CHECKING, TypeVar
import pytest
from trio._tests.pytest_plugin import RUN_SLOW
def check_sequence_matches(seq: Sequence[T], template: Iterable[T | set[T]]) -> None:
    i = 0
    for pattern in template:
        if not isinstance(pattern, set):
            pattern = {pattern}
        got = set(seq[i:i + len(pattern)])
        assert got == pattern
        i += len(got)