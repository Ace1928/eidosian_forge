import codecs
import os
import os.path
import shutil
import string
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import URLError
import numpy as np
import torch
from PIL import Image
from .utils import _flip_byte_order, check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
from .vision import VisionDataset
@property
def labels_file(self) -> str:
    _, (url, _) = self.resources[self.subsets[self.what]]
    return os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0])