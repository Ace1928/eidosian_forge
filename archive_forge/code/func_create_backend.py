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
@classmethod
def create_backend(cls, device_type: str):
    return cls(device_type)