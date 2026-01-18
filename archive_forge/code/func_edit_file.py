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
def edit_file(self, old_path, new_path):
    """Use this tool to edit a file.

        A temporary copy will be edited, and the new contents will be
        returned.

        :return: The new contents of the file.
        """
    old_abs_path, new_abs_path = self._prepare_files(old_path, new_path, allow_write_new=True, force_temp=True)
    command = self._get_command(old_abs_path, new_abs_path)
    subprocess.call(command, cwd=self._root)
    with open(new_abs_path, 'rb') as new_file:
        return new_file.read()