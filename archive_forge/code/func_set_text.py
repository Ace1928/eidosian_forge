from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
import six
from prompt_toolkit.selection import SelectionType
def set_text(self, text):
    """
        Shortcut for setting plain text on clipboard.
        """
    assert isinstance(text, six.string_types)
    self.set_data(ClipboardData(text))