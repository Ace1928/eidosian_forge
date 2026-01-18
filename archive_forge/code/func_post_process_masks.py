import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def post_process_masks(self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None, return_tensors='pt'):
    """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray], List[tf.Tensor]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            reshaped_input_sizes (`Union[torch.Tensor, tf.Tensor, List[Tuple[int,int]]]`):
                The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
            return_tensors (`str`, *optional*, defaults to `"pt"`):
                If `"pt"`, return PyTorch tensors. If `"tf"`, return TensorFlow tensors.
        Returns:
            (`Union[torch.Tensor, tf.Tensor]`): Batched masks in batch_size, num_channels, height, width) format, where
            (height, width) is given by original_size.
        """
    if return_tensors == 'pt':
        return self._post_process_masks_pt(masks=masks, original_sizes=original_sizes, reshaped_input_sizes=reshaped_input_sizes, mask_threshold=mask_threshold, binarize=binarize, pad_size=pad_size)
    elif return_tensors == 'tf':
        return self._post_process_masks_tf(masks=masks, original_sizes=original_sizes, reshaped_input_sizes=reshaped_input_sizes, mask_threshold=mask_threshold, binarize=binarize, pad_size=pad_size)
    else:
        raise ValueError("return_tensors must be either 'pt' or 'tf'")