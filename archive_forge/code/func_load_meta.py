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
def load_meta(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a model meta.json from a path and validate its contents.

    path (Union[str, Path]): Path to meta.json.
    RETURNS (Dict[str, Any]): The loaded meta.
    """
    path = ensure_path(path)
    if not path.parent.exists():
        raise IOError(Errors.E052.format(path=path.parent))
    if not path.exists() or not path.is_file():
        raise IOError(Errors.E053.format(path=path.parent, name='meta.json'))
    meta = srsly.read_json(path)
    for setting in ['lang', 'name', 'version']:
        if setting not in meta or not meta[setting]:
            raise ValueError(Errors.E054.format(setting=setting))
    if 'spacy_version' in meta:
        if not is_compatible_version(about.__version__, meta['spacy_version']):
            lower_version = get_model_lower_version(meta['spacy_version'])
            lower_version = get_base_version(lower_version)
            if lower_version is not None:
                lower_version = 'v' + lower_version
            elif 'spacy_git_version' in meta:
                lower_version = 'git commit ' + meta['spacy_git_version']
            else:
                lower_version = 'version unknown'
            warn_msg = Warnings.W095.format(model=f'{meta['lang']}_{meta['name']}', model_version=meta['version'], version=lower_version, current=about.__version__)
            warnings.warn(warn_msg)
        if is_unconstrained_version(meta['spacy_version']):
            warn_msg = Warnings.W094.format(model=f'{meta['lang']}_{meta['name']}', model_version=meta['version'], version=meta['spacy_version'], example=get_minor_version_range(about.__version__))
            warnings.warn(warn_msg)
    return meta