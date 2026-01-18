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
def save_params(self, filename):
    """[Deprecated] Please use save_parameters. Note that if you want load
        from SymbolBlock later, please use export instead.

        Save parameters to file.

        filename : str
            Path to file.
        """
    warnings.warn('save_params is deprecated. Please use save_parameters. Note that if you want load from SymbolBlock later, please use export instead. For details, see https://mxnet.apache.org/tutorials/gluon/save_load_params.html')
    try:
        self.collect_params().save(filename, strip_prefix=self.prefix)
    except ValueError as e:
        raise ValueError('%s\nsave_params is deprecated. Using save_parameters may resolve this error.' % e.message)