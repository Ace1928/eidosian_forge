from __future__ import annotations
import collections
import configparser
import copy
import os
import os.path
import re
from typing import (
from coverage.exceptions import ConfigError
from coverage.misc import isolate_module, human_sorted_items, substitute_variables
from coverage.tomlconfig import TomlConfigParser, TomlDecodeError
from coverage.types import (
def post_process_file(self, path: str) -> str:
    """Make final adjustments to a file path to make it usable."""
    return os.path.expanduser(path)