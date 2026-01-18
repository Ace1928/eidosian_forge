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
def build_sdist(self, sdist_directory, config_settings=None):
    return self._build_with_temp_dir(['sdist', '--formats', 'gztar'], '.tar.gz', sdist_directory, config_settings)