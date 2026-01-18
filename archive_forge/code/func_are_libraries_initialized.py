import os
import platform
import subprocess
import sys
from shutil import which
from typing import List
import torch
def are_libraries_initialized(*library_names: str) -> List[str]:
    """
    Checks if any of `library_names` are imported in the environment. Will return any names that are.
    """
    return [lib_name for lib_name in library_names if lib_name in sys.modules.keys()]