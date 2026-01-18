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
def from_trees_options(klass, old_tree, new_tree, to_file, path_encoding, external_diff_options, old_label, new_label, using, context_lines):
    """Factory for producing a DiffTree.

        Designed to accept options used by show_diff_trees.

        :param old_tree: The tree to show as old in the comparison
        :param new_tree: The tree to show as new in the comparison
        :param to_file: File to write comparisons to
        :param path_encoding: Character encoding to use for writing paths
        :param external_diff_options: If supplied, use the installed diff
            binary to perform file comparison, using supplied options.
        :param old_label: Prefix to use for old file labels
        :param new_label: Prefix to use for new file labels
        :param using: Commandline to use to invoke an external diff tool
        """
    if using is not None:
        extra_factories = [DiffFromTool.make_from_diff_tree(using, external_diff_options)]
    else:
        extra_factories = []
    if external_diff_options:
        opts = external_diff_options.split()

        def diff_file(olab, olines, nlab, nlines, to_file, path_encoding=None, context_lines=None):
            """:param path_encoding: not used but required
                        to match the signature of internal_diff.
                """
            external_diff(olab, olines, nlab, nlines, to_file, opts)
    else:
        diff_file = internal_diff
    diff_text = DiffText(old_tree, new_tree, to_file, path_encoding, old_label, new_label, diff_file, context_lines=context_lines)
    return klass(old_tree, new_tree, to_file, path_encoding, diff_text, extra_factories)