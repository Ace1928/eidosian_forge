import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@classmethod
def get_binary_path(cls, executable: str):
    """
        Searches for a binary named `executable` in the current PATH. If an executable is found, its absolute path is returned
        else None.
        :param executable: Name of the binary
        :return: Absolute path or None
        """
    if 'PATH' not in os.environ:
        return None
    for directory in os.environ['PATH'].split(get_variable_separator()):
        binary = os.path.abspath(os.path.join(directory, executable))
        if os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary
    return None