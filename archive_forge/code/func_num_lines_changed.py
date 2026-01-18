from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
@property
def num_lines_changed(self):
    return self._num_lines_changed