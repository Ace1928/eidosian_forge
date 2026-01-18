import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def get_rlib_rpath(r_home: str) -> str:
    """Get the path for the R shared library/libraries."""
    lib_path = os.path.join(r_home, get_r_libnn(r_home))
    return lib_path