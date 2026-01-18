from collections import OrderedDict, defaultdict
import warnings
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def list_ctx(self):
    """Returns a list of all the contexts on which the underlying Parameters
        are initialized."""
    s = set()
    for i in self.values():
        s.update(i.list_ctx())
    return list(s)