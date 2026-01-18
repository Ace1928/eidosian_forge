from __future__ import annotations
import typing as t
from jupyter_client.manager import KernelManager
from nbclient.client import NotebookClient
from nbclient.client import execute as _execute
from nbclient.exceptions import CellExecutionError  # noqa: F401
from nbformat import NotebookNode
from .base import Preprocessor
def executenb(*args, **kwargs):
    """DEPRECATED."""
    from warnings import warn
    warn("The 'nbconvert.preprocessors.execute.executenb' function was moved to nbclient.execute. We recommend importing that library directly.", FutureWarning, stacklevel=2)
    return _execute(*args, **kwargs)