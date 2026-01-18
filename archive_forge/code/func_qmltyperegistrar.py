import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def qmltyperegistrar():
    qt_tool_wrapper('qmltyperegistrar', sys.argv[1:], True)