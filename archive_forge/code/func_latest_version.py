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
def latest_version() -> str:
    """Windows version hardcoded to 4.0."""
    if platform.system() == 'Windows':
        return '4.0'
    return ''