import numpy as np
from typing import Union, Tuple, Any, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
def same_padding_transpose_after_stride(strided_size: Tuple[int, int], kernel: Tuple[int, int], stride: Union[int, Tuple[int, int]]) -> (Union[int, Tuple[int, int]], Tuple[int, int]):
    """Computes padding and output size such that TF Conv2DTranspose `same` is matched.

    Note that when padding="same", TensorFlow's Conv2DTranspose makes sure that
    0-padding is added to the already strided image in such a way that the output image
    has the same size as the input image times the stride (and no matter the
    kernel size).

    For example: Input image is (4, 4, 24) (not yet strided), padding is "same",
    stride=2, kernel=5.

    First, the input image is strided (with stride=2):

    Input image (4x4):
    A B C D
    E F G H
    I J K L
    M N O P

    Stride with stride=2 -> (7x7)
    A 0 B 0 C 0 D
    0 0 0 0 0 0 0
    E 0 F 0 G 0 H
    0 0 0 0 0 0 0
    I 0 J 0 K 0 L
    0 0 0 0 0 0 0
    M 0 N 0 O 0 P

    Then this strided image (strided_size=7x7) is padded (exact padding values will be
    output by this function):

    padding -> (left=3, right=2, top=3, bottom=2)

    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 A 0 B 0 C 0 D 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 E 0 F 0 G 0 H 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 I 0 J 0 K 0 L 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 M 0 N 0 O 0 P 0 0
    0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0

    Then deconvolution with kernel=5 yields an output image of 8x8 (x num output
    filters).

    Args:
        strided_size: The size (width x height) of the already strided image.
        kernel: Either width x height (tuple of ints) or - if a square kernel is used -
            a single int for both width and height.
        stride: Either stride width x stride height (tuple of ints) or - if square
            striding is used - a single int for both width- and height striding.

    Returns:
        Tuple consisting of 1) `padding`: A 4-tuple to pad the input after(!) striding.
        The values are for left, right, top, and bottom padding, individually.
        This 4-tuple can be used in a torch.nn.ZeroPad2d layer, and 2) the output shape
        after striding, padding, and the conv transpose layer.
    """
    k_w, k_h = (kernel, kernel) if isinstance(kernel, int) else kernel
    s_w, s_h = (stride, stride) if isinstance(stride, int) else stride
    pad_total_w, pad_total_h = (k_w - 1 + s_w - 1, k_h - 1 + s_h - 1)
    pad_right = pad_total_w // 2
    pad_left = pad_right + (1 if pad_total_w % 2 == 1 else 0)
    pad_bottom = pad_total_h // 2
    pad_top = pad_bottom + (1 if pad_total_h % 2 == 1 else 0)
    output_shape = (strided_size[0] + pad_total_w - k_w + 1, strided_size[1] + pad_total_h - k_h + 1)
    return ((pad_left, pad_right, pad_top, pad_bottom), output_shape)