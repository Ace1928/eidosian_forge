from __future__ import unicode_literals
from prompt_toolkit.buffer import EditReadOnlyBuffer
from prompt_toolkit.filters.cli import ViNavigationMode
from prompt_toolkit.keys import Keys, Key
from prompt_toolkit.utils import Event
from .registry import BaseRegistry
from collections import deque
from six.moves import range
import weakref
import six
@property
def input_processor(self):
    return self._input_processor_ref()