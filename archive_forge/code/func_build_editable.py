import io
import os
import shlex
import sys
import tokenize
import shutil
import contextlib
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
import setuptools
import distutils
from . import errors
from ._path import same_path
from ._reqs import parse_strings
from .warnings import SetuptoolsDeprecationWarning
from distutils.util import strtobool
def build_editable(self, wheel_directory, config_settings=None, metadata_directory=None):
    info_dir = self._get_dist_info_dir(metadata_directory)
    opts = ['--dist-info-dir', info_dir] if info_dir else []
    cmd = ['editable_wheel', *opts, *self._editable_args(config_settings)]
    with suppress_known_deprecation():
        return self._build_with_temp_dir(cmd, '.whl', wheel_directory, config_settings)