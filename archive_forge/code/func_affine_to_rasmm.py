import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
@affine_to_rasmm.setter
def affine_to_rasmm(self, value):
    if value is not None:
        value = np.array(value)
        if value.shape != (4, 4):
            msg = f'Affine matrix has a shape of (4, 4) but a ndarray with shape {value.shape} was provided instead.'
            raise ValueError(msg)
    self._affine_to_rasmm = value