import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_cupy_install_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))