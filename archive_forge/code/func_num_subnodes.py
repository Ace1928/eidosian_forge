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
def num_subnodes(self):
    """Get the number of subnodes to iterate in this MapNode"""
    self._get_inputs()
    self._check_iterfield()
    if self._serial:
        return 1
    if self.nested:
        return len(ensure_list(flatten(getattr(self.inputs, self.iterfield[0]))))
    return len(ensure_list(getattr(self.inputs, self.iterfield[0])))