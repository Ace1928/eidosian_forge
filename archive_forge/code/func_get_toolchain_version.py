import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from collections import OrderedDict
from time import sleep
from typing import Any, Dict, List
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import pushd, validate_dir, wrap_url_progress_hook
def get_toolchain_version(name: str, version: str) -> str:
    """Toolchain version."""
    toolchain_folder = ''
    if platform.system() == 'Windows':
        toolchain_folder = '{}{}'.format(name, version.replace('.', ''))
    return toolchain_folder