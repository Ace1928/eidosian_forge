import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def getArray(self, attr, roffset, coffset, rows, cols, format):
    name = attr.split('\t')[-1]
    array = pydevd_vars.eval_in_context(name, self.get_namespace(), self.get_namespace())
    return pydevd_vars.table_like_struct_to_xml(array, name, roffset, coffset, rows, cols, format)