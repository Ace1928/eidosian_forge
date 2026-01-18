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
def prepare_metadata_for_build_wheel(self, metadata_directory, config_settings=None):
    sys.argv = [*sys.argv[:1], *self._global_args(config_settings), 'dist_info', '--output-dir', metadata_directory, '--keep-egg-info']
    with no_install_setup_requires():
        self.run_setup()
    self._bubble_up_info_directory(metadata_directory, '.egg-info')
    return self._bubble_up_info_directory(metadata_directory, '.dist-info')