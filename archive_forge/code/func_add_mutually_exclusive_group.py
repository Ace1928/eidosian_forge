import copy as _copy
import os as _os
import re as _re
import sys as _sys
import textwrap as _textwrap
from gettext import gettext as _
def add_mutually_exclusive_group(self, **kwargs):
    group = _MutuallyExclusiveGroup(self, **kwargs)
    self._mutually_exclusive_groups.append(group)
    return group