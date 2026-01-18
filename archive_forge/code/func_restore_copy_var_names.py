import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def restore_copy_var_names(blocks, save_copies, typemap):
    """
    restores variable names of user variables after applying copy propagation
    """
    if not save_copies:
        return {}
    rename_dict = {}
    var_rename_map = {}
    for a, b in save_copies:
        if not a.startswith('$') and b.name.startswith('$') and (b.name not in rename_dict):
            new_name = mk_unique_var('${}'.format(a))
            rename_dict[b.name] = new_name
            var_rename_map[new_name] = a
            typ = typemap.pop(b.name)
            typemap[new_name] = typ
    replace_var_names(blocks, rename_dict)
    return var_rename_map