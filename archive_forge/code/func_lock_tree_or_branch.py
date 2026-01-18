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
def lock_tree_or_branch(wt, br):
    if wt is not None:
        exit_stack.enter_context(wt.lock_read())
    elif br is not None:
        exit_stack.enter_context(br.lock_read())