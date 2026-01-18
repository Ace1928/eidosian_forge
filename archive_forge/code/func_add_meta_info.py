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
def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
    """
        Custom the ir, module, metadata and asm per backend specific requirement
        """
    raise NotImplementedError