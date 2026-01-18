from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from pymatgen.core.tensors import Tensor
@classmethod
def from_vasp_voigt(cls, input_vasp_array: ArrayLike) -> Self:
    """
        Args:
            input_vasp_array (nd.array): Voigt form of tensor.

        Returns:
            PiezoTensor
        """
    voigt_map = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    input_vasp_array = np.array(input_vasp_array)
    rank = 3
    pt = np.zeros([rank, 3, 3])
    for dim in range(rank):
        for pos, val in enumerate(voigt_map):
            pt[dim][val] = input_vasp_array[dim][pos]
            pt[dim].T[val] = input_vasp_array[dim][pos]
    return cls(pt)