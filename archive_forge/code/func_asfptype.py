import numpy
import cupy
from cupy import _core
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def asfptype(self):
    """Upcasts matrix to a floating point format.

        When the matrix has floating point type, the method returns itself.
        Otherwise it makes a copy with floating point type and the same format.

        Returns:
            cupyx.scipy.sparse.spmatrix: A matrix with float type.

        """
    if self.dtype.kind == 'f':
        return self
    else:
        typ = numpy.promote_types(self.dtype, 'f')
        return self.astype(typ)