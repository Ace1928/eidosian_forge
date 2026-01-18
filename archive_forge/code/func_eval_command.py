from __future__ import absolute_import
import os
import re
import sys
import trace
import inspect
import warnings
import unittest
import textwrap
import tempfile
import functools
import traceback
import itertools
import gdb
from .. import libcython
from .. import libpython
from . import TestLibCython as test_libcython
from ...Utils import add_metaclass
def eval_command(self, command):
    gdb.execute('cy exec open(%r, "w").write(str(%s))' % (self.tmpfilename, command))
    return self.tmpfile.read().strip()