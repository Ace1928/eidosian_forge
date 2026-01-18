from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
@property
def mem_gb(self):
    """Get estimated memory (GB)"""
    if hasattr(self._interface, 'estimated_memory_gb'):
        self._mem_gb = self._interface.estimated_memory_gb
        logger.warning('Setting "estimated_memory_gb" on Interfaces has been deprecated as of nipype 1.0, please use Node.mem_gb.')
    return self._mem_gb