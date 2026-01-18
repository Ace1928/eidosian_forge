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
def load_metadata(repo_id, class_info_file):
    fname = os.path.join('' if repo_id is None else repo_id, class_info_file)
    if not os.path.exists(fname) or not os.path.isfile(fname):
        if repo_id is None:
            raise ValueError(f'Could not file {fname} locally. repo_id must be defined if loading from the hub')
        try:
            fname = hf_hub_download(repo_id, class_info_file, repo_type='dataset')
        except RepositoryNotFoundError:
            fname = hf_hub_download(repo_id, class_info_file)
    with open(fname, 'r') as f:
        class_info = json.load(f)
    return class_info