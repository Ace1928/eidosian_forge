import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def getpreferredencoding() -> str:
    """Get the user's preferred encoding."""
    return locale.getpreferredencoding() or sys.getdefaultencoding()