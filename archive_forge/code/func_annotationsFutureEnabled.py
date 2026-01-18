import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
@annotationsFutureEnabled.setter
def annotationsFutureEnabled(self, value):
    assert value is True
    assert isinstance(self.scope, ModuleScope)
    self.scope._annotations_future_enabled = True