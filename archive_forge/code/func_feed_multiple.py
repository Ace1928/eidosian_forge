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
def feed_multiple(self, key_presses: list[KeyPress], first: bool=False) -> None:
    """
        :param first: If true, insert before everything else.
        """
    if first:
        self.input_queue.extendleft(reversed(key_presses))
    else:
        self.input_queue.extend(key_presses)