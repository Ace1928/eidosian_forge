from tests.unit import AWSMockServiceTestCase
from boto.s3.connection import S3Connection
from boto.s3.bucket import Bucket
from boto.s3.lifecycle import Rule, Lifecycle, Transition
def test_expiration_is_optional(self):
    t = Transition(days=30, storage_class='GLACIER')
    r = Rule('myid', 'prefix', 'Enabled', expiration=None, transition=t)
    xml = r.to_xml()
    self.assertIn('<Transition><StorageClass>GLACIER</StorageClass><Days>30</Days>', xml)