from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
@property
def effective_width(self):
    """The effective width when the indentation level is considered."""
    return self._console_width - INDENTATION_WIDTH * self._level