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
def get_device_properties(self, device):
    raise NotImplementedError