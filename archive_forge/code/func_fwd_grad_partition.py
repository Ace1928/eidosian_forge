from . import numpy_wrapper as anp
from .numpy_vjps import (untake, balanced_eq, match_complex, replace_zero,
from autograd.extend import (defjvp, defjvp_argnum, def_linear, vspace, JVPNode,
from ..util import func
from .numpy_boxes import ArrayBox
def fwd_grad_partition(g, ans, x, kth, axis=-1, kind='introselect', order=None):
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return g[partition_perm]