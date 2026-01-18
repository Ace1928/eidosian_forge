from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
def make_abc_branch(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\n'))], revision_id=b'A')
    builder.build_snapshot([b'A'], [('add', ('dir', b'dir-id', 'directory', None))], revision_id=b'B')
    builder.build_snapshot([b'B'], [('modify', ('file', b'new content\n'))], revision_id=b'C')
    builder.finish_series()
    return builder.get_branch()