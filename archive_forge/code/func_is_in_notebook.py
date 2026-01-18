import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
def is_in_notebook():
    try:
        get_ipython = sys.modules['IPython'].get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            raise ImportError('console')
        if 'VSCODE_PID' in os.environ:
            raise ImportError('vscode')
        if 'DATABRICKS_RUNTIME_VERSION' in os.environ and os.environ['DATABRICKS_RUNTIME_VERSION'] < '11.0':
            raise ImportError('databricks')
        return importlib.util.find_spec('IPython') is not None
    except (AttributeError, ImportError, KeyError):
        return False