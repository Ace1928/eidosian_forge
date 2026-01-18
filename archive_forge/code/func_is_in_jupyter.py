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
def is_in_jupyter() -> bool:
    """Check if user is running spaCy from a Jupyter or Colab notebook by
    detecting the IPython kernel. Mainly used for the displaCy visualizer.
    RETURNS (bool): True if in Jupyter/Colab, False if not.
    """
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True
        if get_ipython().__class__.__module__ == 'google.colab._shell':
            return True
    except NameError:
        pass
    try:
        import google.colab
        return True
    except ImportError:
        pass
    return False