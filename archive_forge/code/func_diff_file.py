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
def diff_file(olab, olines, nlab, nlines, to_file, path_encoding=None, context_lines=None):
    """:param path_encoding: not used but required
                        to match the signature of internal_diff.
                """
    external_diff(olab, olines, nlab, nlines, to_file, opts)