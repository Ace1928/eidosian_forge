import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
def supports_box_chars() -> bool:
    """Check if the encoding supports Unicode box characters."""
    return all(map(can_encode, '│─└┘┌┐'))