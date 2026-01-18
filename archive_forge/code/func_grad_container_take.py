import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
def grad_container_take(ans, A, idx):
    return lambda g: container_untake(g, idx, vspace(A))