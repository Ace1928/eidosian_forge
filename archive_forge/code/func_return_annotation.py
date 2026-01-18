from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types
from funcsigs.version import __version__
@property
def return_annotation(self):
    return self._return_annotation