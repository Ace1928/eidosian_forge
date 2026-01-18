import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
def is_valid_annotation_coco_detection(annotation: Dict[str, Union[List, Tuple]]) -> bool:
    if isinstance(annotation, dict) and 'image_id' in annotation and ('annotations' in annotation) and isinstance(annotation['annotations'], (list, tuple)) and (len(annotation['annotations']) == 0 or isinstance(annotation['annotations'][0], dict)):
        return True
    return False