import boto.swf.layer2
from boto.swf.layer2 import SWFBase
from tests.unit import unittest
from mock import Mock
def test_instantiation(self):
    self.assertEquals(MOCK_DOMAIN, self.swf_base.domain)
    self.assertEquals(MOCK_ACCESS_KEY, self.swf_base.aws_access_key_id)
    self.assertEquals(MOCK_SECRET_KEY, self.swf_base.aws_secret_access_key)
    self.assertEquals(MOCK_REGION, self.swf_base.region)
    boto.swf.layer2.Layer1.assert_called_with(MOCK_ACCESS_KEY, MOCK_SECRET_KEY, region=MOCK_REGION)