import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
def fwd_grad_make_sequence(argnum, g, ans, seq_type, *args, **kwargs):
    return container_untake(g, argnum - 1, vspace(ans))