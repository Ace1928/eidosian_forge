from __future__ import unicode_literals
from .buffer import Buffer, AcceptAction
from .buffer_mapping import BufferMapping
from .clipboard import Clipboard, InMemoryClipboard
from .enums import DEFAULT_BUFFER, EditingMode
from .filters import CLIFilter, to_cli_filter
from .key_binding.bindings.basic import load_basic_bindings
from .key_binding.bindings.emacs import load_emacs_bindings
from .key_binding.bindings.vi import load_vi_bindings
from .key_binding.registry import BaseRegistry
from .key_binding.defaults import load_key_bindings
from .layout import Window
from .layout.containers import Container
from .layout.controls import BufferControl
from .styles import DEFAULT_STYLE, Style
import six
def dummy_handler(cli):
    """ Dummy event handler. """