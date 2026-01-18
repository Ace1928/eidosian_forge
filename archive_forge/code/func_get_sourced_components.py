import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
def get_sourced_components(config: Union[Dict[str, Any], Config]) -> Dict[str, Dict[str, Any]]:
    """RETURNS (List[str]): All sourced components in the original config,
    e.g. {"source": "en_core_web_sm"}. If the config contains a key
    "factory", we assume it refers to a component factory.
    """
    return {name: cfg for name, cfg in config.get('components', {}).items() if 'factory' not in cfg and 'source' in cfg}