from subprocess import check_output
import os.path
from collections import defaultdict
import inspect
from functools import partial
import numba
from numba.core.registry import cpu_target
from all overloads.
def gen_lower_listing(path=None):
    """
    Generate lowering listing to ``path`` or (if None) to stdout.
    """
    cpu_backend = cpu_target.target_context
    cpu_backend.refresh()
    fninfos = gather_function_info(cpu_backend)
    out = format_function_infos(fninfos)
    if path is None:
        print(out)
    else:
        with open(path, 'w') as fobj:
            print(out, file=fobj)