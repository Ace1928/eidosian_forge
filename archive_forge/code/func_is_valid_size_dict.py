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
def is_valid_size_dict(size_dict):
    if not isinstance(size_dict, dict):
        return False
    size_dict_keys = set(size_dict.keys())
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    return False