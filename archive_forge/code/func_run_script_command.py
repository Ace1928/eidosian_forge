from __future__ import annotations
from . import _pathlib
import sys
import os.path
import platform
import importlib
import argparse
import typing as T
from .utils.core import MesonException, MesonBugException
from . import mlog
def run_script_command(script_name: str, script_args: T.List[str]) -> int:
    script_map = {'exe': 'meson_exe', 'install': 'meson_install', 'delsuffix': 'delwithsuffix', 'gtkdoc': 'gtkdochelper', 'hotdoc': 'hotdochelper', 'regencheck': 'regen_checker'}
    module_name = script_map.get(script_name, script_name)
    try:
        module = importlib.import_module('mesonbuild.scripts.' + module_name)
    except ModuleNotFoundError as e:
        mlog.exception(e)
        return 1
    try:
        return module.run(script_args)
    except MesonException as e:
        mlog.error(f'Error in {script_name} helper script:')
        mlog.exception(e)
        return 1