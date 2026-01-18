from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
def input_filter(self, keys: list[str], raw: list[int]) -> list[str]:
    """
        This function is passed each all the input events and raw keystroke
        values. These values are passed to the *input_filter* function
        passed to the constructor. That function must return a list of keys to
        be passed to the widgets to handle. If no *input_filter* was
        defined this implementation will return all the input events.
        """
    if self._input_filter:
        return self._input_filter(keys, raw)
    return keys