import collections
import os
from xml.etree.ElementTree import Element as ET_Element
from .vision import VisionDataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
@property
def masks(self) -> List[str]:
    return self.targets