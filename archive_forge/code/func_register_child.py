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
def register_child(self, block, name=None):
    if not isinstance(block, HybridBlock):
        raise ValueError('Children of HybridBlock must also be HybridBlock, but %s has type %s. If you are using Sequential, please try HybridSequential instead.' % (str(block), str(type(block))))
    super(HybridBlock, self).register_child(block, name)
    self._clear_cached_op()