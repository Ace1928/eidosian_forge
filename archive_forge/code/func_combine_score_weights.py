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
def combine_score_weights(weights: List[Dict[str, Optional[float]]], overrides: Dict[str, Optional[float]]=SimpleFrozenDict()) -> Dict[str, Optional[float]]:
    """Combine and normalize score weights defined by components, e.g.
    {"ents_r": 0.2, "ents_p": 0.3, "ents_f": 0.5} and {"some_other_score": 1.0}.

    weights (List[dict]): The weights defined by the components.
    overrides (Dict[str, Optional[Union[float, int]]]): Existing scores that
        should be preserved.
    RETURNS (Dict[str, float]): The combined and normalized weights.
    """
    result: Dict[str, Optional[float]] = {key: value for w_dict in weights for key, value in w_dict.items()}
    result.update(overrides)
    weight_sum = sum([v if v else 0.0 for v in result.values()])
    for key, value in result.items():
        if value and weight_sum > 0:
            result[key] = round(value / weight_sum, 2)
    return result