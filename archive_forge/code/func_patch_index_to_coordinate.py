import copy
import math
import re
from typing import List, Optional, Tuple, Union
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType
def patch_index_to_coordinate(ul_idx: int, lr_idx: int, num_patches_per_side: int):
    """
    Given a grid of length `num_patches_per_side` and the indices of the upper-left and lower-right corners of a
    bounding box, returns the normalized coordinates of the bounding box, in the form (x1, y1, x2, y2).

    Args:
        ul_idx (`int`): the index of the grid cell that corresponds to the upper-left corner of the bounding box.
        lr_idx (`int`): the index of the grid cell that corresponds to the lower-right corner of the bounding box.
        num_patches_per_side (`int`): the number of patches along each side.

    Returns:
        `Tuple[float]`: the normalized coordinates of the bounding box, in the form (x1, y1, x2, y2).
    """
    cell_size = 1.0 / num_patches_per_side
    ul_x = ul_idx % num_patches_per_side
    ul_y = ul_idx // num_patches_per_side
    lr_x = lr_idx % num_patches_per_side
    lr_y = lr_idx // num_patches_per_side
    if ul_idx == lr_idx:
        x1 = ul_x * cell_size
        y1 = ul_y * cell_size
        x2 = lr_x * cell_size + cell_size
        y2 = lr_y * cell_size + cell_size
    elif ul_x == lr_x or ul_y == lr_y:
        x1 = ul_x * cell_size
        y1 = ul_y * cell_size
        x2 = lr_x * cell_size + cell_size
        y2 = lr_y * cell_size + cell_size
    else:
        x1 = ul_x * cell_size + cell_size / 2
        y1 = ul_y * cell_size + cell_size / 2
        x2 = lr_x * cell_size + cell_size / 2
        y2 = lr_y * cell_size + cell_size / 2
    return (x1, y1, x2, y2)