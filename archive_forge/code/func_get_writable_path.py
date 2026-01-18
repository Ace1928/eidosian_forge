import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
def get_writable_path(path: str) -> str:
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))