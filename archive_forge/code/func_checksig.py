from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def checksig(*args, **kwargs):
    sig.bind(*args, **kwargs)