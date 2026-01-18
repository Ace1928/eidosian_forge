from __future__ import annotations
import weakref
from asyncio import Task, sleep
from collections import deque
from typing import TYPE_CHECKING, Any, Generator
from prompt_toolkit.application.current import get_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters.app import vi_navigation_mode
from prompt_toolkit.keys import Keys
from prompt_toolkit.utils import Event
from .key_bindings import Binding, KeyBindingsBase
def empty_queue(self) -> list[KeyPress]:
    """
        Empty the input queue. Return the unprocessed input.
        """
    key_presses = list(self.input_queue)
    self.input_queue.clear()
    key_presses = [k for k in key_presses if k.key != Keys.CPRResponse]
    return key_presses