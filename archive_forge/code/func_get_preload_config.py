import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def get_preload_config() -> Optional[Dict[str, Any]]:
    global _preload_config
    if _preload_config is None:
        _preload_config = _get_json_data('_wheel.json')
    return _preload_config