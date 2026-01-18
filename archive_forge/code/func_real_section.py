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
def real_section(self, section: str) -> str | None:
    """Get the actual name of a section."""
    for section_prefix in self.section_prefixes:
        real_section = section_prefix + section
        has = super().has_section(real_section)
        if has:
            return real_section
    return None