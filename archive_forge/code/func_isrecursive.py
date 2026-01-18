import collections as _collections
import dataclasses as _dataclasses
import re
import sys as _sys
import types as _types
from io import StringIO as _StringIO
def isrecursive(self, object):
    return self.format(object, {}, 0, 0)[2]