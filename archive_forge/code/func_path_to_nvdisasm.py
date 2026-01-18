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
@functools.lru_cache()
def path_to_nvdisasm():
    return _path_to_binary('nvdisasm')