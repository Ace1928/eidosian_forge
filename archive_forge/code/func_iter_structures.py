from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
def iter_structures(self):
    """
        Iterates over all brain structures in the order that they appear along the axis

        Yields
        ------
        tuple with 3 elements:
        - CIFTI-2 brain structure name
        - slice to select the data associated with the brain structure from the tensor
        - brain model covering that specific brain structure
        """
    idx_start = 0
    start_name = self.name[idx_start]
    for idx_current, name in enumerate(self.name):
        if start_name != name:
            yield (start_name, slice(idx_start, idx_current), self[idx_start:idx_current])
            idx_start = idx_current
            start_name = self.name[idx_start]
    yield (start_name, slice(idx_start, None), self[idx_start:])