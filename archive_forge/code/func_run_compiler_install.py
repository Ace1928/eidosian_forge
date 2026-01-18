import argparse
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
from cmdstanpy.utils.cmdstan import get_download_url
from . import progress as progbar
def run_compiler_install(dir: str, verbose: bool, progress: bool) -> None:
    from .install_cxx_toolchain import is_installed as _is_installed_cxx
    from .install_cxx_toolchain import run_rtools_install as _main_cxx
    from .utils import cxx_toolchain_path
    compiler_found = False
    rtools40_home = os.environ.get('RTOOLS40_HOME')
    for cxx_loc in ([rtools40_home] if rtools40_home is not None else []) + [home_cmdstan(), os.path.join(os.path.abspath('/'), 'RTools40'), os.path.join(os.path.abspath('/'), 'RTools'), os.path.join(os.path.abspath('/'), 'RTools35'), os.path.join(os.path.abspath('/'), 'RBuildTools')]:
        for cxx_version in ['40', '35']:
            if _is_installed_cxx(cxx_loc, cxx_version):
                compiler_found = True
                break
        if compiler_found:
            break
    if not compiler_found:
        print('Installing RTools40')
        _main_cxx({'dir': dir, 'progress': progress, 'version': None, 'verbose': verbose})
        cxx_version = '40'
    cxx_toolchain_path(cxx_version, dir)