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
def get_subnodes(self):
    """Generate subnodes of a mapnode and write pre-execution report"""
    self._get_inputs()
    self._check_iterfield()
    write_node_report(self, result=None, is_mapnode=True)
    return [node for _, node in self._make_nodes()]