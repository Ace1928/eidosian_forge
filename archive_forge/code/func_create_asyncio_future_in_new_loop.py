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
def create_asyncio_future_in_new_loop() -> asyncio.Future[object]:
    with closing(asyncio.new_event_loop()) as loop:
        return loop.create_future()