from __future__ import unicode_literals
import functools
import os
import signal
import six
import sys
import textwrap
import threading
import time
import types
import weakref
from subprocess import Popen
from .application import Application, AbortAction
from .buffer import Buffer
from .buffer_mapping import BufferMapping
from .completion import CompleteEvent, get_common_complete_suffix
from .enums import SEARCH_BUFFER
from .eventloop.base import EventLoop
from .eventloop.callbacks import EventLoopCallbacks
from .filters import Condition
from .input import StdinInput, Input
from .key_binding.input_processor import InputProcessor
from .key_binding.input_processor import KeyPress
from .key_binding.registry import Registry
from .key_binding.vi_state import ViState
from .keys import Keys
from .output import Output
from .renderer import Renderer, print_tokens
from .search_state import SearchState
from .utils import Event
from .buffer import AcceptAction
def run_application_generator(self, coroutine, render_cli_done=False):
    """
        EXPERIMENTAL
        Like `run_in_terminal`, but takes a generator that can yield Application instances.

        Example:

            def f():
                yield Application1(...)
                print('...')
                yield Application2(...)
            cli.run_in_terminal_async(f)

        The values which are yielded by the given coroutine are supposed to be
        `Application` instances that run in the current CLI, all other code is
        supposed to be CPU bound, so except for yielding the applications,
        there should not be any user interaction or I/O in the given function.
        """
    if render_cli_done:
        self._return_value = True
        self._redraw()
        self.renderer.reset()
    else:
        self.renderer.erase()
    self._return_value = None
    g = coroutine()
    assert isinstance(g, types.GeneratorType)

    def step_next(send_value=None):
        """ Execute next step of the coroutine."""
        try:
            with self.input.cooked_mode():
                result = g.send(send_value)
        except StopIteration:
            done()
        except:
            done()
            raise
        else:
            assert isinstance(result, Application)
            self.run_sub_application(result, done_callback=step_next, _from_application_generator=True)

    def done():
        self.renderer.reset()
        self.renderer.request_absolute_cursor_position()
        self._redraw()
    step_next()