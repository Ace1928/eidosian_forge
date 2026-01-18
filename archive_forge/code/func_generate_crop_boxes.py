import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def generate_crop_boxes(self, image, target_size, crop_n_layers: int=0, overlap_ratio: float=512 / 1500, points_per_crop: Optional[int]=32, crop_n_points_downscale_factor: Optional[List[int]]=1, device: Optional['torch.device']=None, input_data_format: Optional[Union[str, ChannelDimension]]=None, return_tensors: str='pt'):
    """
        Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`np.array`):
                Input original image
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
                each layer has 2**i_layer number of image crops.
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sample from each crop.
            crop_n_points_downscale_factor (`List[int]`, *optional*, defaults to 1):
                The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            device (`torch.device`, *optional*, defaults to None):
                Device to use for the computation. If None, cpu will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`. If `tf`, returns `tf.Tensor`.
        """
    crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(image, target_size, crop_n_layers, overlap_ratio, points_per_crop, crop_n_points_downscale_factor, input_data_format)
    if return_tensors == 'pt':
        if device is None:
            device = torch.device('cpu')
        crop_boxes = torch.tensor(crop_boxes, device=device)
        points_per_crop = torch.tensor(points_per_crop, device=device)
        input_labels = torch.tensor(input_labels, device=device)
    elif return_tensors == 'tf':
        if device is not None:
            raise ValueError('device is not a supported argument when return_tensors is tf!')
        crop_boxes = tf.convert_to_tensor(crop_boxes)
        points_per_crop = tf.convert_to_tensor(points_per_crop)
        input_labels = tf.convert_to_tensor(input_labels)
    else:
        raise ValueError("return_tensors must be either 'pt' or 'tf'.")
    return (crop_boxes, points_per_crop, cropped_images, input_labels)