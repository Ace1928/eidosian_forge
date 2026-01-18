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
def replace_model_node(model: Model, target: Model, replacement: Model) -> None:
    """Replace a node within a model with a new one, updating refs.

    model (Model): The parent model.
    target (Model): The target node.
    replacement (Model): The node to replace the target with.
    """
    for node in model.walk():
        if target in node.layers:
            node.layers[node.layers.index(target)] = replacement
    for node in model.walk():
        for ref_name in node.ref_names:
            if node.maybe_get_ref(ref_name) is target:
                node.set_ref(ref_name, replacement)