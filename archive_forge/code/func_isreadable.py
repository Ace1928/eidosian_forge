import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def isreadable(self, object):
    s, readable, recursive = self.format(object, {}, 0, 0)
    return readable and (not recursive)