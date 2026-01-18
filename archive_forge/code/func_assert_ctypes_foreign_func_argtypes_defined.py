import ast
import os
import re
from hacking import core
from os_win.utils.winapi import libs as w_lib
import_translation_for_log_or_exception = re.compile(
@core.flake8ext
def assert_ctypes_foreign_func_argtypes_defined(logical_line):
    res = ctypes_func_typedefs_re.findall(logical_line)
    for lib_name, func_name in res:
        mod_path = '%s.py' % os.path.join(os.path.dirname(w_lib.__file__), lib_name)
        module_src = _get_module_src(mod_path)
        argtypes_expr = '%s.argtypes =' % func_name
        restype_expr = '%s.restype =' % func_name
        if not (argtypes_expr in module_src and restype_expr in module_src):
            yield (0, 'O302: Foreign function called using ctypes without having its argument and return value types declared in %s.%s.py.' % (w_lib.__name__, lib_name))