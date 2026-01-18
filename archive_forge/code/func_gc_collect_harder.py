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
def gc_collect_harder() -> None:
    for _ in range(5):
        gc.collect()