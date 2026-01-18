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
@n_procs.setter
def n_procs(self, value):
    """Set an estimated number of processes/threads"""
    self._n_procs = value
    if hasattr(self._interface.inputs, 'num_threads'):
        self._interface.inputs.num_threads = self._n_procs