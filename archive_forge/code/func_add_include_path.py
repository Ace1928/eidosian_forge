import io
import json
import os
import platform
import shutil
import subprocess
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from cmdstanpy.utils import get_logger
from cmdstanpy.utils.cmdstan import (
from cmdstanpy.utils.command import do_command
from cmdstanpy.utils.filesystem import SanitizedOrTmpFilePath
def add_include_path(self, path: str) -> None:
    """Adds include path to existing set of compiler options."""
    path = os.path.abspath(os.path.expanduser(path))
    if 'include-paths' not in self._stanc_options:
        self._stanc_options['include-paths'] = [path]
    elif path not in self._stanc_options['include-paths']:
        self._stanc_options['include-paths'].append(path)