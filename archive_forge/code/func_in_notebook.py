import importlib
import logging
import sys
import textwrap
from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from packaging.version import Version
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import DeveloperAPI
from ray.widgets import Template
@DeveloperAPI
def in_notebook(shell_name: Optional[str]=None) -> bool:
    """Return whether we are in a Jupyter notebook or qtconsole."""
    if not shell_name:
        shell_name = _get_ipython_shell_name()
    return shell_name == 'ZMQInteractiveShell'