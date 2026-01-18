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
def show_diff_trees(old_tree, new_tree, to_file, specific_files=None, external_diff_options=None, old_label: str='a/', new_label: str='b/', extra_trees=None, path_encoding: str='utf8', using: Optional[str]=None, format_cls=None, context=DEFAULT_CONTEXT_AMOUNT):
    """Show in text form the changes from one tree to another.

    :param to_file: The output stream.
    :param specific_files: Include only changes to these files - None for all
        changes.
    :param external_diff_options: If set, use an external GNU diff and pass
        these options.
    :param extra_trees: If set, more Trees to use for looking up file ids
    :param path_encoding: If set, the path will be encoded as specified,
        otherwise is supposed to be utf8
    :param format_cls: Formatter class (DiffTree subclass)
    """
    if context is None:
        context = DEFAULT_CONTEXT_AMOUNT
    if format_cls is None:
        format_cls = DiffTree
    with contextlib.ExitStack() as exit_stack:
        exit_stack.enter_context(old_tree.lock_read())
        if extra_trees is not None:
            for tree in extra_trees:
                exit_stack.enter_context(tree.lock_read())
        exit_stack.enter_context(new_tree.lock_read())
        differ = format_cls.from_trees_options(old_tree, new_tree, to_file, path_encoding, external_diff_options, old_label, new_label, using, context_lines=context)
        return differ.show_diff(specific_files, extra_trees)