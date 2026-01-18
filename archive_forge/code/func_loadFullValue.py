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
@silence_warnings_decorator
def loadFullValue(self, seq, scope_attrs):
    """
        Evaluate full value for async Console variables in a separate thread and send results to IDE side
        :param seq: id of command
        :param scope_attrs: a sequence of variables with their attributes separated by NEXT_VALUE_SEPARATOR
        (i.e.: obj	attr1	attr2NEXT_VALUE_SEPARATORobj2\x07ttr1	attr2)
        :return:
        """
    frame_variables = self.get_namespace()
    var_objects = []
    vars = scope_attrs.split(NEXT_VALUE_SEPARATOR)
    for var_attrs in vars:
        if '\t' in var_attrs:
            name, attrs = var_attrs.split('\t', 1)
        else:
            name = var_attrs
            attrs = None
        if name in frame_variables:
            var_object = pydevd_vars.resolve_var_object(frame_variables[name], attrs)
            var_objects.append((var_object, name))
        else:
            var_object = pydevd_vars.eval_in_context(name, frame_variables, frame_variables)
            var_objects.append((var_object, name))
    from _pydevd_bundle.pydevd_comm import GetValueAsyncThreadConsole
    py_db = getattr(self, 'debugger', None)
    if py_db is None:
        py_db = get_global_debugger()
    if py_db is None:
        from pydevd import PyDB
        py_db = PyDB()
    t = GetValueAsyncThreadConsole(py_db, self.get_server(), seq, var_objects)
    t.start()