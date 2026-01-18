import cupy
from cupyx import cusparse
from cupy_backends.cuda.api import driver
from cupy_backends.cuda.api import runtime
import cupyx.scipy.sparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
def sort_indices(self):
    """Sorts the indices of this matrix *in place*.

        .. warning::
            Calling this function might synchronize the device.

        """
    if not self.has_sorted_indices:
        cusparse.cscsort(self)
        self.has_sorted_indices = True