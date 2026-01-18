import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_cub_path():
    global _cub_path
    if _cub_path == '':
        _cub_path = _get_cub_path()
    return _cub_path