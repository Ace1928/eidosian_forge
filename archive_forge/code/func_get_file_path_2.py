import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
def get_file_path_2(*path_components: str) -> str:
    return os.path.join(*path_components)