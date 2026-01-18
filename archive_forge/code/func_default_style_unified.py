import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def default_style_unified(diff_opts):
    """Default to unified diff style if alternative not specified in diff_opts.

        diff only allows one style to be specified; they don't override.
        Note that some of these take optargs, and the optargs can be
        directly appended to the options.
        This is only an approximate parser; it doesn't properly understand
        the grammar.

    :param diff_opts: List of options for external (GNU) diff.
    :return: List of options with default style=='unified'.
    """
    for s in style_option_list:
        for j in diff_opts:
            if j.startswith(s):
                break
        else:
            continue
        break
    else:
        diff_opts.append('-u')
    return diff_opts