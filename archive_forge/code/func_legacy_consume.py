from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def legacy_consume(self, kwargs):
    """Consume arguments from the root of the configuration dictionary.
    """
    self.consume_known(kwargs)
    for descr in self._field_registry:
        if isinstance(descr, SubtreeDescriptor):
            descr.legacy_shim_consume(self, kwargs)