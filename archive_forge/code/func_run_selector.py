from __future__ import annotations
import asyncio
import os
import select
import selectors
import sys
import threading
from asyncio import AbstractEventLoop, get_running_loop
from selectors import BaseSelector, SelectorKey
from typing import TYPE_CHECKING, Any, Callable, Mapping
def run_selector() -> None:
    nonlocal ready, result
    result = self.selector.select(timeout=timeout)
    os.write(self._w, b'x')
    ready = True