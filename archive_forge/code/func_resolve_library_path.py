import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
def resolve_library_path(path: str) -> str:
    return os.path.realpath(path)