import copy
import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature as BaseBatchFeature
from .image_transforms import center_crop, normalize, rescale
from .image_utils import ChannelDimension
from .utils import (
def convert_to_size_dict(size, max_size: Optional[int]=None, default_to_square: bool=True, height_width_order: bool=True):
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError('Cannot specify both size as an int, with default_to_square=True and max_size')
        return {'height': size, 'width': size}
    elif isinstance(size, int) and (not default_to_square):
        size_dict = {'shortest_edge': size}
        if max_size is not None:
            size_dict['longest_edge'] = max_size
        return size_dict
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {'height': size[0], 'width': size[1]}
    elif isinstance(size, (tuple, list)) and (not height_width_order):
        return {'height': size[1], 'width': size[0]}
    elif size is None and max_size is not None:
        if default_to_square:
            raise ValueError('Cannot specify both default_to_square=True and max_size')
        return {'longest_edge': max_size}
    raise ValueError(f'Could not convert size input to size dict: {size}')