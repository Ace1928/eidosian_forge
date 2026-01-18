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
def is_exec_available(cls, executable: str) -> bool:
    return cls.get_binary_path(executable) is not None