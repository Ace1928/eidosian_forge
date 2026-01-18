from typing import Any
import numpy as np
from ray.air.util.tensor_extensions.utils import create_ragged_ndarray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _truncated_repr
def convert_udf_returns_to_numpy(udf_return_col: Any) -> Any:
    """Convert UDF columns (output of map_batches) to numpy, if possible.

    This includes lists of scalars, objects supporting the array protocol, and lists
    of objects supporting the array protocol, such as `[1, 2, 3]`, `Tensor([1, 2, 3])`,
    and `[array(1), array(2), array(3)]`.

    Returns:
        The input as an np.ndarray if possible, otherwise the original input.

    Raises:
        ValueError if an input was array-like but we failed to convert it to an array.
    """
    if isinstance(udf_return_col, np.ndarray):
        return udf_return_col
    if isinstance(udf_return_col, list):
        if len(udf_return_col) == 1 and isinstance(udf_return_col[0], np.ndarray):
            udf_return_col = np.expand_dims(udf_return_col[0], axis=0)
            return udf_return_col
        try:
            if all((is_valid_udf_return(e) and (not is_scalar_list(e)) for e in udf_return_col)):
                udf_return_col = [np.asarray(e) for e in udf_return_col]
            shapes = set()
            has_object = False
            for e in udf_return_col:
                if isinstance(e, np.ndarray):
                    shapes.add((e.dtype, e.shape))
                elif isinstance(e, bytes):
                    has_object = True
                elif not np.isscalar(e):
                    has_object = True
            if has_object or len(shapes) > 1:
                udf_return_col = create_ragged_ndarray(udf_return_col)
            else:
                udf_return_col = np.array(udf_return_col)
        except Exception as e:
            raise ValueError(f'Failed to convert column values to numpy array: ({_truncated_repr(udf_return_col)}): {e}.')
    elif hasattr(udf_return_col, '__array__'):
        try:
            udf_return_col = np.array(udf_return_col)
        except Exception as e:
            raise ValueError(f'Failed to convert column values to numpy array: ({_truncated_repr(udf_return_col)}): {e}.')
    return udf_return_col