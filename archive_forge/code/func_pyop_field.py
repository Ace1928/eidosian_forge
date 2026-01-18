from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def pyop_field(self, name):
    """
        Get a PyObjectPtr for the given PyObject* field within this PyObject,
        coping with some python 2 versus python 3 differences.
        """
    return PyObjectPtr.from_pyobject_ptr(self.field(name))