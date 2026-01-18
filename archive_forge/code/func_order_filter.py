import warnings
import cupy
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.signal import _signaltools_core as _st_core
def order_filter(a, domain, rank):
    """Perform an order filter on an N-D array.

    Perform an order filter on the array in. The domain argument acts as a mask
    centered over each pixel. The non-zero elements of domain are used to
    select elements surrounding each input pixel which are placed in a list.
    The list is sorted, and the output for that pixel is the element
    corresponding to rank in the sorted list.

    Args:
        a (cupy.ndarray): The N-dimensional input array.
        domain (cupy.ndarray): A mask array with the same number of dimensions
            as `a`. Each dimension should have an odd number of elements.
        rank (int): A non-negative integer which selects the element from the
            sorted list (0 corresponds to the smallest element).

    Returns:
        cupy.ndarray: The results of the order filter in an array with the same
        shape as `a`.

    .. seealso:: :func:`cupyx.scipy.ndimage.rank_filter`
    .. seealso:: :func:`scipy.signal.order_filter`
    """
    if a.dtype.kind in 'bc' or a.dtype == cupy.float16:
        raise ValueError('data type not supported')
    if any((x % 2 != 1 for x in domain.shape)):
        raise ValueError('Each dimension of domain argument  should have an odd number of elements.')
    return _filters.rank_filter(a, rank, footprint=domain, mode='constant')