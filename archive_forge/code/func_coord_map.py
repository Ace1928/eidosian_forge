import numpy as np
from scipy import ndimage as ndi
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform
from ._warps_cy import _warp_fast
from ..measure import block_reduce
from .._shared.utils import (
def coord_map(*args):
    return inverse_map(*args, **map_args)