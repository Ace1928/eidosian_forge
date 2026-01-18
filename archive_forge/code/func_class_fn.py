from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
@classmethod
def class_fn(cls, args):
    return args + cls.CLASS_STATE