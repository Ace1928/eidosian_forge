import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def rcc():
    args = []
    user_args = sys.argv[1:]
    if '--binary' not in user_args:
        args.extend(['-g', 'python'])
    args.extend(user_args)
    qt_tool_wrapper('rcc', args, True)