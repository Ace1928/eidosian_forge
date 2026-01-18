from __future__ import print_function
import asyncio
import base64
import errno
from getpass import getpass
from io import BytesIO
import os
from queue import Empty
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from warnings import warn
from typing import Dict as DictType, Any as AnyType
from zmq import ZMQError
from IPython.core import page
from traitlets import (
from traitlets.config import SingletonConfigurable
from .completer import ZMQCompleter
from .zmqhistory import ZMQHistoryManager
from . import __version__
from prompt_toolkit import __version__ as ptk_version
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import (
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.shortcuts import print_formatted_text, CompleteStyle
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.layout.processors import (
from prompt_toolkit.styles import merge_styles
from prompt_toolkit.styles.pygments import (style_from_pygments_cls,
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.utils import suspend_to_background_supported
from pygments.styles import get_style_by_name
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound
from pygments.token import Token
from jupyter_console.utils import run_sync, ensure_async
def handle_image(self, data, mime):
    handler = getattr(self, 'handle_image_{0}'.format(self.image_handler), None)
    if handler:
        return handler(data, mime)