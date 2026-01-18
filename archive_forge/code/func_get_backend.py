import functools
import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import traceback
from typing import Dict
from ..runtime.driver import DriverBase
def get_backend(device_type: str):
    if device_type not in _backends:
        device_backend_package_name = f'...third_party.{device_type}'
        if importlib.util.find_spec(device_backend_package_name, package=__spec__.name):
            try:
                importlib.import_module(device_backend_package_name, package=__spec__.name)
            except Exception:
                traceback.print_exc()
        else:
            return None
    return _backends[device_type] if device_type in _backends else None