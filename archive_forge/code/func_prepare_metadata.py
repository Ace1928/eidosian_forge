import json
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def prepare_metadata(class_info):
    metadata = {}
    class_names = []
    thing_ids = []
    for key, info in class_info.items():
        metadata[key] = info['name']
        class_names.append(info['name'])
        if info['isthing']:
            thing_ids.append(int(key))
    metadata['thing_ids'] = thing_ids
    metadata['class_names'] = class_names
    return metadata