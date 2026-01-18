import os
import logging
from typing import Optional, List, Tuple
from functools import lru_cache
from importlib.util import find_spec
from ray._private.accelerators.accelerator import AcceleratorManager
@lru_cache()
def is_package_present(package_name: str) -> bool:
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False