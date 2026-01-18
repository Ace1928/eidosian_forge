import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def pyside_script_wrapper(script_name):
    """Launch a script shipped with PySide."""
    script = Path(__file__).resolve().parent / script_name
    command = [sys.executable, os.fspath(script)] + sys.argv[1:]
    sys.exit(subprocess.call(command))