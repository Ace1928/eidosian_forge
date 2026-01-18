from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
def matching_names(self):
    """Field name equals value."""
    Point = collections.namedtuple('Point', ['x', 'y'])
    return Point(x='x', y='y')