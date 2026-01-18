from __future__ import absolute_import
import sys
import types
def is_basestring(t):
    """Return true if t is (referentially) the abstract basestring."""
    return t is basestring