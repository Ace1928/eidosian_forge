import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
def get_env_var(name: str, default: Optional[str]=None, *, namespaces: Tuple=('LANGSMITH', 'LANGCHAIN')) -> Optional[str]:
    """Retrieve an environment variable from a list of namespaces.

    Args:
        name (str): The name of the environment variable.
        default (Optional[str], optional): The default value to return if the
            environment variable is not found. Defaults to None.
        namespaces (Tuple, optional): A tuple of namespaces to search for the
            environment variable. Defaults to ("LANGSMITH", "LANGCHAINs").

    Returns:
        Optional[str]: The value of the environment variable if found,
            otherwise the default value.
    """
    names = [f'{namespace}_{name}' for namespace in namespaces]
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default