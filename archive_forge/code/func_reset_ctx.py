import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def reset_ctx(self, ctx):
    """Re-assign all Parameters to other contexts. If the Block is hybridized, it will reset the _cached_op_args.
        Parameters
        ----------
        ctx : Context or list of Context, default :py:meth:`context.current_context()`.
            Assign Parameter to given context. If ctx is a list of Context, a
            copy will be made for each context.
        """
    params = self.collect_params()
    if self._cached_op:
        for p in self._cached_op_args:
            if p.name not in params:
                p.reset_ctx(ctx)
    for p in params.values():
        p.reset_ctx(ctx)