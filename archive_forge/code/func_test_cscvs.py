from ....revision import Revision
from ....tests import TestCase
from ..pseudonyms import extract_foreign_revids
def test_cscvs(self):
    x = Revision(b'myrevid')
    x.properties = {'cscvs-svn-repository-uuid': 'someuuid', 'cscvs-svn-revision-number': '4', 'cscvs-svn-branch-path': '/trunk'}
    self.assertEqual({('svn', 'someuuid:4:trunk')}, extract_foreign_revids(x))