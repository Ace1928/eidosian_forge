import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def genpyi():
    pyside_dir = Path(__file__).resolve().parents[1]
    support = pyside_dir / 'support'
    cmd = support / 'generate_pyi.py'
    command = [sys.executable, os.fspath(cmd)] + sys.argv[1:]
    sys.exit(subprocess.call(command))