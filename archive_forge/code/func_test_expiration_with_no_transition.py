from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_expiration_with_no_transition(self):
    lifecycle = Lifecycle()
    lifecycle.add_rule('myid', 'prefix', 'Enabled', 30)
    xml = lifecycle.to_xml()
    self.assertIn('<Expiration><Days>30</Days></Expiration>', xml)