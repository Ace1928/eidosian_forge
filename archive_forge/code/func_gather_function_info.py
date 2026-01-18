from subprocess import check_output
import os.path
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target
from all overloads.
def gather_function_info(backend):
    fninfos = defaultdict(list)
    basepath = os.path.dirname(os.path.dirname(numba.__file__))
    for fn, osel in backend._defns.items():
        for sig, impl in osel.versions:
            info = {}
            fninfos[fn].append(info)
            info['fn'] = fn
            info['sig'] = sig
            code, firstlineno = inspect.getsourcelines(impl)
            path = inspect.getsourcefile(impl)
            info['impl'] = {'name': get_func_name(impl), 'filename': os.path.relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
    return fninfos