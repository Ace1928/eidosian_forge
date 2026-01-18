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
def test_get_set_default_format(self):
    old_default = controldir.format_registry.get('default')
    old_default_help = controldir.format_registry.get_help('default')
    private_default = old_default().repository_format.__class__
    old_format = repository.format_registry.get_default()
    self.assertTrue(isinstance(old_format, private_default))

    def make_sample_bzrdir():
        my_bzrdir = bzrdir.BzrDirMetaFormat1()
        my_bzrdir.repository_format = SampleRepositoryFormat()
        return my_bzrdir
    controldir.format_registry.remove('default')
    controldir.format_registry.register('sample', make_sample_bzrdir, '')
    controldir.format_registry.set_default('sample')
    try:
        dir = bzrdir.BzrDirMetaFormat1().initialize('memory:///')
        result = dir.create_repository()
        self.assertEqual(result, 'A bzr repository dir')
    finally:
        controldir.format_registry.remove('default')
        controldir.format_registry.remove('sample')
        controldir.format_registry.register('default', old_default, old_default_help)
    self.assertIsInstance(repository.format_registry.get_default(), old_format.__class__)