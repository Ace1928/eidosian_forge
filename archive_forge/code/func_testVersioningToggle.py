from xml import sax
from boto import handler
from boto.gs import acl
from tests.integration.gs.testcase import GSTestCase
def testVersioningToggle(self):
    b = self._MakeBucket()
    self.assertFalse(b.get_versioning_status())
    b.configure_versioning(True)
    self.assertTrue(b.get_versioning_status())
    b.configure_versioning(False)
    self.assertFalse(b.get_versioning_status())