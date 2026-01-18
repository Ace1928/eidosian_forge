import ast
import logging
import os
import re
import sys
import warnings
from typing import List
from importlib import util
from importlib.metadata import version
from pathlib import Path
from . import Nuitka, run_command
def is_installed(self, package):
    return bool(util.find_spec(package))