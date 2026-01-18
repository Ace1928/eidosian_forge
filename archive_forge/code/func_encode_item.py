from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def encode_item(item):
    if six.PY2 and isinstance(item, unicode):
        return item.encode('utf-8')
    else:
        return item