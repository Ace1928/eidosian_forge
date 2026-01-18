from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.console import console_attr
import six
@property
def completer_class(self):
    return self._completer_class