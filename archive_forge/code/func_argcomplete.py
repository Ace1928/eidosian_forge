from __future__ import annotations
import argparse
import errno
import json
import os
import site
import sys
import sysconfig
from pathlib import Path
from shutil import which
from subprocess import Popen
from typing import Any
from . import paths
from .version import __version__
def argcomplete(self) -> None:
    """Trigger auto-completion, if enabled"""
    try:
        import argcomplete
        argcomplete.autocomplete(self)
    except ImportError:
        pass