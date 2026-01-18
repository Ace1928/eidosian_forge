import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
def set_destination_filename(self, default):
    if self.target is not None:
        self._fname = self.target
    else:
        self._fname = envvar.PYOMO_CONFIG_DIR
        if not os.path.isdir(self._fname):
            os.makedirs(self._fname)
    if os.path.isdir(self._fname):
        self._fname = os.path.join(self._fname, default)
    targetDir = os.path.dirname(self._fname)
    if not os.path.isdir(targetDir):
        os.makedirs(targetDir)