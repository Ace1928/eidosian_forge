import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def is_virtual_env():
    return sys.prefix != sys.base_prefix