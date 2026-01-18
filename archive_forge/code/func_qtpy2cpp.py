import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def qtpy2cpp():
    pyside_script_wrapper('qtpy2cpp.py')