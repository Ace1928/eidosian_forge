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
def normalize_version(version: str) -> str:
    """Return maj.min part of version string."""
    if platform.system() == 'Windows':
        if version in ['4', '40']:
            version = '4.0'
        elif version == '35':
            version = '3.5'
    return version