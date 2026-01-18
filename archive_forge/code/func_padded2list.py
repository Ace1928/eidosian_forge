import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def padded2list(self, padded: Padded) -> List2d:
    """Unpack a Padded datatype to a list of 2-dimensional arrays."""
    data = padded.data
    indices = to_numpy(padded.indices)
    lengths = to_numpy(padded.lengths)
    unpadded: List[Optional[Array2d]] = [None] * len(lengths)
    data = self.as_contig(data.transpose((1, 0, 2)))
    for i in range(data.shape[0]):
        unpadded[indices[i]] = data[i, :int(lengths[i])]
    return cast(List2d, unpadded)