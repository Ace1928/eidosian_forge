import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
def make_diverged_master_branch(self):
    """
        B: wt.branch.last_revision()
        M: wt.branch.get_master_branch().last_revision()
        W: wt.last_revision()


            1
            |          B-2 3
            | |
            4 5-M
            |
            W
        """
    format = self.workingtree_format.get_controldir_for_branch()
    builder = self.make_branch_builder('.', format=format)
    builder.start_series()
    revids = {}
    revids['1'] = builder.build_snapshot(None, [('add', ('', None, 'directory', '')), ('add', ('file1', None, 'file', b'file1 content\n'))])
    revids['2'] = builder.build_snapshot([revids['1']], [])
    revids['4'] = builder.build_snapshot([revids['1']], [('add', ('file4', None, 'file', b'file4 content\n'))])
    revids['3'] = builder.build_snapshot([revids['1']], [])
    revids['5'] = builder.build_snapshot([revids['3']], [('add', ('file5', None, 'file', b'file5 content\n'))])
    builder.finish_series()
    return (builder, builder._branch.last_revision(), revids)