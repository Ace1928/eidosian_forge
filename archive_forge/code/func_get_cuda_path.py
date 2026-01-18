import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_cuda_path():
    global _cuda_path
    if _cuda_path == '':
        _cuda_path = _get_cuda_path()
    return _cuda_path