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
def test_find_format(self):
    self.build_tree(['foo/', 'bar/'])

    def check_format(format, url):
        dir = format._matchingcontroldir.initialize(url)
        format.initialize(dir)
        found_format = bzrrepository.RepositoryFormatMetaDir.find_format(dir)
        self.assertIsInstance(found_format, format.__class__)
    check_format(repository.format_registry.get_default(), 'bar')