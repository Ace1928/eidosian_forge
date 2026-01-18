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
@classmethod
def make_from_diff_tree(klass, command_string, external_diff_options=None):

    def from_diff_tree(diff_tree):
        full_command_string = [command_string]
        if external_diff_options is not None:
            full_command_string.extend(external_diff_options.split())
        return klass.from_string(full_command_string, diff_tree.old_tree, diff_tree.new_tree, diff_tree.to_file)
    return from_diff_tree