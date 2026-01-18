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
def append_to_arg_count(self, data):
    """
        Add digit to the input argument.

        :param data: the typed digit as string
        """
    assert data in '-0123456789'
    current = self._arg
    if data == '-':
        assert current is None or current == '-'
        result = data
    elif current is None:
        result = data
    else:
        result = '%s%s' % (current, data)
    self.input_processor.arg = result