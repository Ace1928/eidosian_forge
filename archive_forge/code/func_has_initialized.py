from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import subprocess
import importlib
import pkg_resources
import threading
from subprocess import check_output
from dataclasses import dataclass
from typing import Optional
from fileio import File, PathIO, PathIOLike
from lazyops.envs import logger
from lazyops.envs import LazyEnv
@property
def has_initialized(self):
    return bool(self.submodule_name in sys.modules)