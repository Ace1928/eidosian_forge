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
def stdout_proxy(self, raw=False):
    """
        Create an :class:`_StdoutProxy` class which can be used as a patch for
        `sys.stdout`. Writing to this proxy will make sure that the text
        appears above the prompt, and that it doesn't destroy the output from
        the renderer.

        :param raw: (`bool`) When True, vt100 terminal escape sequences are not
                    removed/escaped.
        """
    return _StdoutProxy(self, raw=raw)