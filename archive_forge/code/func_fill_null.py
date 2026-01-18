from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def fill_null(values, fill_value):
    """Replace each null element in values with a corresponding
    element from fill_value.

    If fill_value is scalar-like, then every null element in values
    will be replaced with fill_value. If fill_value is array-like,
    then the i-th element in values will be replaced with the i-th
    element in fill_value.

    The fill_value's type must be the same as that of values, or it
    must be able to be implicitly casted to the array's type.

    This is an alias for :func:`coalesce`.

    Parameters
    ----------
    values : Array, ChunkedArray, or Scalar-like object
        Each null element is replaced with the corresponding value
        from fill_value.
    fill_value : Array, ChunkedArray, or Scalar-like object
        If not same type as values, will attempt to cast.

    Returns
    -------
    result : depends on inputs
        Values with all null elements replaced

    Examples
    --------
    >>> import pyarrow as pa
    >>> arr = pa.array([1, 2, None, 3], type=pa.int8())
    >>> fill_value = pa.scalar(5, type=pa.int8())
    >>> arr.fill_null(fill_value)
    <pyarrow.lib.Int8Array object at ...>
    [
      1,
      2,
      5,
      3
    ]
    >>> arr = pa.array([1, 2, None, 4, None])
    >>> arr.fill_null(pa.array([10, 20, 30, 40, 50]))
    <pyarrow.lib.Int64Array object at ...>
    [
      1,
      2,
      30,
      4,
      50
    ]
    """
    if not isinstance(fill_value, (pa.Array, pa.ChunkedArray, pa.Scalar)):
        fill_value = pa.scalar(fill_value, type=values.type)
    elif values.type != fill_value.type:
        fill_value = pa.scalar(fill_value.as_py(), type=values.type)
    return call_function('coalesce', [values, fill_value])