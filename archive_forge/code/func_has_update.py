from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
@property
def has_update(self):
    return self._has_update