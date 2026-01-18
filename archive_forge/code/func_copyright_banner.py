import argparse
from typing import Tuple, List, Optional, NoReturn, Callable
import code
import curtsies
import cwcwidth
import greenlet
import importlib.util
import logging
import os
import pygments
import requests
import sys
import xdg
from pathlib import Path
from . import __version__, __copyright__
from .config import default_config_path, Config
from .translations import _
def copyright_banner() -> str:
    return _('{} See AUTHORS.rst for details.').format(__copyright__)