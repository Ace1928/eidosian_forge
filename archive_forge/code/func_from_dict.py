import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
@classmethod
def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
    """
        Overrides the `from_dict` method from the base class to make sure `do_reduce_labels` is updated if image
        processor is created using from_dict and kwargs e.g. `SegformerImageProcessor.from_pretrained(checkpoint,
        reduce_labels=True)`
        """
    image_processor_dict = image_processor_dict.copy()
    if 'reduce_labels' in kwargs:
        image_processor_dict['reduce_labels'] = kwargs.pop('reduce_labels')
    return super().from_dict(image_processor_dict, **kwargs)