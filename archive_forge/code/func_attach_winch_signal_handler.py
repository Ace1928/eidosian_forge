from __future__ import annotations
import asyncio
import contextvars
import os
import re
import signal
import sys
import threading
import time
from asyncio import (
from contextlib import ExitStack, contextmanager
from subprocess import Popen
from traceback import format_tb
from typing import (
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.clipboard import Clipboard, InMemoryClipboard
from prompt_toolkit.cursor_shapes import AnyCursorShapeConfig, to_cursor_shape_config
from prompt_toolkit.data_structures import Size
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.eventloop import (
from prompt_toolkit.eventloop.utils import call_soon_threadsafe
from prompt_toolkit.filters import Condition, Filter, FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.input.typeahead import get_typeahead, store_typeahead
from prompt_toolkit.key_binding.bindings.page_navigation import (
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.emacs_state import EmacsState
from prompt_toolkit.key_binding.key_bindings import (
from prompt_toolkit.key_binding.key_processor import KeyPressEvent, KeyProcessor
from prompt_toolkit.key_binding.vi_state import ViState
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import Container, Window
from prompt_toolkit.layout.controls import BufferControl, UIControl
from prompt_toolkit.layout.dummy import create_dummy_layout
from prompt_toolkit.layout.layout import Layout, walk
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.renderer import Renderer, print_formatted_text
from prompt_toolkit.search import SearchState
from prompt_toolkit.styles import (
from prompt_toolkit.utils import Event, in_main_thread
from .current import get_app_session, set_app
from .run_in_terminal import in_terminal, run_in_terminal
@contextmanager
def attach_winch_signal_handler(handler: Callable[[], None]) -> Generator[None, None, None]:
    """
    Attach the given callback as a WINCH signal handler within the context
    manager. Restore the original signal handler when done.

    The `Application.run` method will register SIGWINCH, so that it will
    properly repaint when the terminal window resizes. However, using
    `run_in_terminal`, we can temporarily send an application to the
    background, and run an other app in between, which will then overwrite the
    SIGWINCH. This is why it's important to restore the handler when the app
    terminates.
    """
    sigwinch = getattr(signal, 'SIGWINCH', None)
    if sigwinch is None or not in_main_thread():
        yield
        return
    loop = get_running_loop()
    previous_winch_handler = getattr(loop, '_signal_handlers', {}).get(sigwinch)
    try:
        loop.add_signal_handler(sigwinch, handler)
        yield
    finally:
        loop.remove_signal_handler(sigwinch)
        if previous_winch_handler is not None:
            loop.add_signal_handler(sigwinch, previous_winch_handler._callback, *previous_winch_handler._args)