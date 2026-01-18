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
def resolve_dot_names(config: Config, dot_names: List[Optional[str]]) -> Tuple[Any, ...]:
    """Resolve one or more "dot notation" names, e.g. corpora.train.
    The paths could point anywhere into the config, so we don't know which
    top-level section we'll be looking within.

    We resolve the whole top-level section, although we could resolve less --
    we could find the lowest part of the tree.
    """
    resolved = {}
    output: List[Any] = []
    errors = []
    for name in dot_names:
        if name is None:
            output.append(name)
        else:
            section = name.split('.')[0]
            if section not in resolved:
                if registry.is_promise(config[section]):
                    result = registry.resolve({'config': config[section]})['config']
                else:
                    result = registry.resolve(config[section])
                resolved[section] = result
            try:
                output.append(dot_to_object(resolved, name))
            except KeyError:
                msg = f'not a valid section reference: {name}'
                errors.append({'loc': name.split('.'), 'msg': msg})
    if errors:
        raise ConfigValidationError(config=config, errors=errors)
    return tuple(output)