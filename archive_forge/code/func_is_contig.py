from .abstract import ArrayCompatible, Dummy, IterableType, IteratorType
from numba.core.errors import NumbaTypeError, NumbaValueError
@property
def is_contig(self):
    return self.layout in 'CF'