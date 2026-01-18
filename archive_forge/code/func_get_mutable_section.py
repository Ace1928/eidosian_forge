import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def get_mutable_section(self, section_id=None):
    try:
        self.load()
    except transport.NoSuchFile:
        self._load_from_string(b'')
    if section_id in self.dirty_sections:
        return self.dirty_sections[section_id]
    if section_id is None:
        section = self._config_obj
    else:
        section = self._config_obj.setdefault(section_id, {})
    mutable_section = self.mutable_section_class(section_id, section)
    self.dirty_sections[section_id] = mutable_section
    return mutable_section