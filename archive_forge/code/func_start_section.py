import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def start_section(self, heading):
    self._indent()
    section = self._Section(self, self._current_section, heading)
    self._add_item(section.format_help, [])
    self._current_section = section