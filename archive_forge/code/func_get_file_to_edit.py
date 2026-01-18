import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple
from pip._internal.exceptions import (
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum
def get_file_to_edit(self) -> Optional[str]:
    """Returns the file with highest priority in configuration"""
    assert self.load_only is not None, 'Need to be specified a file to be editing'
    try:
        return self._get_parser_to_modify()[0]
    except IndexError:
        return None