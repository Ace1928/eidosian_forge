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
def config_files_to_try(config_file: bool | str) -> list[tuple[str, bool, bool]]:
    """What config files should we try to read?

    Returns a list of tuples:
        (filename, is_our_file, was_file_specified)
    """
    if config_file == '.coveragerc':
        config_file = True
    specified_file = config_file is not True
    if not specified_file:
        rcfile = os.getenv('COVERAGE_RCFILE')
        if rcfile:
            config_file = rcfile
            specified_file = True
    if not specified_file:
        config_file = '.coveragerc'
    assert isinstance(config_file, str)
    files_to_try = [(config_file, True, specified_file), ('setup.cfg', False, False), ('tox.ini', False, False), ('pyproject.toml', False, False)]
    return files_to_try