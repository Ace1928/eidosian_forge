from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import TensorType, logging
def make_batched(videos) -> List[List[ImageInput]]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)):
        return videos
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        videos_dim = np.array(videos[0]).ndim
        if videos_dim == 3:
            return [videos]
        elif videos_dim == 4:
            return videos
    elif is_valid_image(videos):
        videos_dim = np.array(videos).ndim
        if videos_dim == 3:
            return [[videos]]
        elif videos_dim == 4:
            return [videos]
        elif videos_dim == 5:
            return videos
    raise ValueError(f'Could not make batched video from {videos}')