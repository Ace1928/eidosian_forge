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
def resize_annotation(self, annotation, orig_size, size, resample: PILImageResampling=PILImageResampling.NEAREST) -> Dict:
    """
        Resize the annotation to match the resized image. If size is an int, smaller edge of the mask will be matched
        to this number.
        """
    return resize_annotation(annotation, orig_size=orig_size, target_size=size, resample=resample)