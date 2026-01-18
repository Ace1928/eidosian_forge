import io
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import BaseImageProcessor, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def prepare_coco_panoptic(self, *args, **kwargs):
    logger.warning_once('The `prepare_coco_panoptic` method is deprecated and will be removed in v4.33. ')
    return prepare_coco_panoptic_annotation(*args, **kwargs)