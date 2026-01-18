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
@futuresAllowed.setter
def futuresAllowed(self, value):
    assert value is False
    if isinstance(self.scope, ModuleScope):
        self.scope._futures_allowed = False