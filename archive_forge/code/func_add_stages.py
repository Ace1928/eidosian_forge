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
def add_stages(self, arch, extern_libs, stages):
    """
        Custom the arch, extern_libs and stages per backend specific requirement
        """
    raise NotImplementedError