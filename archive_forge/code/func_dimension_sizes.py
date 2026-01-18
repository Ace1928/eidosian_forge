from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload
import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument
import xarray as xr
from xarray.core.types import T_DuckArray
def dimension_sizes(*, dim_names: st.SearchStrategy[Hashable]=names(), min_dims: int=0, max_dims: int=3, min_side: int=1, max_side: Union[int, None]=None) -> st.SearchStrategy[Mapping[Hashable, int]]:
    """
    Generates an arbitrary mapping from dimension names to lengths.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_names: strategy generating strings, optional
        Strategy for generating dimension names.
        Defaults to the `names` strategy.
    min_dims: int, optional
        Minimum number of dimensions in generated list.
        Default is 1.
    max_dims: int, optional
        Maximum number of dimensions in generated list.
        Default is 3.
    min_side: int, optional
        Minimum size of a dimension.
        Default is 1.
    max_side: int, optional
        Minimum size of a dimension.
        Default is `min_length` + 5.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    if max_side is None:
        max_side = min_side + 3
    return st.dictionaries(keys=dim_names, values=st.integers(min_value=min_side, max_value=max_side), min_size=min_dims, max_size=max_dims)