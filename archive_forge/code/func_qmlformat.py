import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def qmlformat():
    qt_tool_wrapper('qmlformat', sys.argv[1:])