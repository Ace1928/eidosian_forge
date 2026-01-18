from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def check_databases(self, t):
    """check knit content for a repository."""
    self.assertHasNoKndx(t, 'inventory')
    self.assertHasNoKnit(t, 'inventory')
    self.assertHasNoKndx(t, 'revisions')
    self.assertHasNoKnit(t, 'revisions')
    self.assertHasNoKndx(t, 'signatures')
    self.assertHasNoKnit(t, 'signatures')
    self.assertFalse(t.has('knits'))
    self.assertEqual([], list(self.index_class(t, 'pack-names', None).iter_all_entries()))
    self.assertTrue(S_ISDIR(t.stat('packs').st_mode))
    self.assertTrue(S_ISDIR(t.stat('upload').st_mode))
    self.assertTrue(S_ISDIR(t.stat('indices').st_mode))
    self.assertTrue(S_ISDIR(t.stat('obsolete_packs').st_mode))