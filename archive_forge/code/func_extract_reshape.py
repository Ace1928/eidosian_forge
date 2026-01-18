from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
def extract_reshape(self, src: np.ndarray, object=True) -> npt.NDArray[Any]:
    """
        Given an array where the final dimension is the flattened output of a
        Stan model, (e.g. one row of a Stan CSV file), extract the variable
        and reshape it to the correct type and dimensions.

        This will most likely result in copies of the data being made if
        the variable is not a scalar.

        Parameters
        ----------
        src : np.ndarray
            The array to extract from.

            Indicies besides the final dimension are preserved
            in the output.

        object : bool
            If True, the output of tuple types will be an object array,
            otherwise it will use custom dtypes to represent tuples.

        Returns
        -------
        npt.NDArray[Any]
            The extracted variable, reshaped to the correct dimensions.
            If the variable is a tuple, this will be an object array,
            otherwise it will have a dtype of either float64 or complex128.
        """
    out = self._extract_helper(src)
    if not object:
        out = out.astype(self.dtype())
    if src.ndim > 1:
        out = out.reshape(*src.shape[:-1], *self.dimensions, order='F')
    else:
        out = out.squeeze(axis=0)
    return out