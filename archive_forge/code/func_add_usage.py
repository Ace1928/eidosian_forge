import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def add_usage(self, usage, actions, groups, prefix=None):
    if usage is not SUPPRESS:
        args = (usage, actions, groups, prefix)
        self._add_item(self._format_usage, args)