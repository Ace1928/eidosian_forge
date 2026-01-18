from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_instance_usage_audit_log_with_before_unicode(self):
    before = '\\u5de5\\u4f5c'
    self.assertRaises(exceptions.BadRequest, self.cs.instance_usage_audit_log.get, before)