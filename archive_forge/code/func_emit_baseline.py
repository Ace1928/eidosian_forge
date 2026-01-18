import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def emit_baseline(self, revobj, ref):
    mark = 1
    self.revid_to_mark[revobj.revision_id] = b'%d' % mark
    tree_old = self.branch.repository.revision_tree(breezy.revision.NULL_REVISION)
    [tree_new] = list(self._get_revision_trees([revobj.revision_id]))
    file_cmds = self._get_filecommands(tree_old, tree_new)
    self.print_cmd(commands.ResetCommand(ref, None))
    self.print_cmd(self._get_commit_command(ref, mark, revobj, file_cmds))