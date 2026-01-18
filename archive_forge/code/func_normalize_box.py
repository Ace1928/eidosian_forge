from typing import Dict, Iterable, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format, to_pil_image
from ...image_utils import (
from ...utils import TensorType, is_pytesseract_available, is_vision_available, logging, requires_backends
def normalize_box(box, width, height):
    return [int(1000 * (box[0] / width)), int(1000 * (box[1] / height)), int(1000 * (box[2] / width)), int(1000 * (box[3] / height))]