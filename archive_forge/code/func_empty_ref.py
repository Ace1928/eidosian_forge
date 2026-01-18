from __future__ import annotations
import numpy as np
import xarray as xr
from .linalg import einsum, einsum_path, matmul
from .accessors import LinAlgAccessor, EinopsAccessor
def empty_ref(*args, dims, dtype=None):
    """Create an empty DataArray from reference object(s).

    Creates an empty DataArray from reference
    DataArrays or Datasets and a list with the desired dimensions.

    Parameters
    ----------
    *args : iterable of DataArray or Dataset
        Reference objects from which the lengths and coordinate values (if any)
        of the given `dims` will be taken.
    dims : list of hashable
        List of dimensions of the output DataArray. Passed as is to the
        {class}`~xarray.DataArray` constructor.
    dtype : dtype, optional
        The dtype of the output array.
        If it is not provided it will be inferred from the reference
        DataArrays in `args` with :func:`numpy.result_type`.

    Returns
    -------
    DataArray

    See Also
    --------
    ones_ref, zeros_ref
    """
    return _create_ref(*args, dims=dims, np_creator=np.empty, dtype=dtype)