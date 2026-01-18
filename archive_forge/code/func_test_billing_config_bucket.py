import os
import re
import urllib
import xml.sax
from six import StringIO
from boto import handler
from boto import storage_uri
from boto.gs.acl import ACL
from boto.gs.cors import Cors
from boto.gs.lifecycle import LifecycleConfig
from tests.integration.gs.testcase import GSTestCase
def test_billing_config_bucket(self):
    """Test setting and getting of billing config on Bucket."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    billing = bucket.get_billing_config()
    self.assertEqual(billing, BILLING_EMPTY)
    bucket.configure_billing(requester_pays=True)
    billing = bucket.get_billing_config()
    self.assertEqual(billing, BILLING_ENABLED)
    bucket.configure_billing(requester_pays=False)
    billing = bucket.get_billing_config()
    self.assertEqual(billing, BILLING_DISABLED)