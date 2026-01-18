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
def submodule_name(self):
    if not self.module:
        return self.name
    if not self.module_path:
        return self.module + '.' + self.name
    if self.name in self.module_path:
        return self.module_path
    return self.module_path + '.' + self.name