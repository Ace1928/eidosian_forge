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
@needed_outputs.setter
def needed_outputs(self, new_outputs):
    """Needed outputs changes the hash, refresh if changed"""
    new_outputs = sorted(list(set(new_outputs or [])))
    if new_outputs != self._needed_outputs:
        self._hashvalue = None
        self._hashed_inputs = None
        self._needed_outputs = new_outputs