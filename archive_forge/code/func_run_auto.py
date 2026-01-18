import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
def run_auto(self, names_list, after, dry_run):
    if names_list is not None and len(names_list) > 1:
        raise errors.CommandError(gettext('Only one path may be specified to --auto.'))
    if after:
        raise errors.CommandError(gettext('--after cannot be specified with --auto.'))
    work_tree, file_list = WorkingTree.open_containing_paths(names_list, default_directory='.')
    self.enter_context(work_tree.lock_tree_write())
    rename_map.RenameMap.guess_renames(work_tree.basis_tree(), work_tree, dry_run)