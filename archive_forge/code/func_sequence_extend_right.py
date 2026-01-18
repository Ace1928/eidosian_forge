import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
@primitive
def sequence_extend_right(seq, *elts):
    return seq + type(seq)(elts)