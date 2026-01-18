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
def prepare_metadata_for_build_editable(self, metadata_directory, config_settings=None):
    return self.prepare_metadata_for_build_wheel(metadata_directory, config_settings)