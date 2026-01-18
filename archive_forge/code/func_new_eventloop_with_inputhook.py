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
def new_eventloop_with_inputhook(inputhook: Callable[[InputHookContext], None]) -> AbstractEventLoop:
    """
    Create a new event loop with the given inputhook.
    """
    selector = InputHookSelector(selectors.DefaultSelector(), inputhook)
    loop = asyncio.SelectorEventLoop(selector)
    return loop