from mock import patch, Mock
import unittest
import time
from boto.exception import S3ResponseError
from boto.s3.connection import S3Connection
from boto.s3.bucketlogging import BucketLogging
from boto.s3.lifecycle import Lifecycle
from boto.s3.lifecycle import Transition
from boto.s3.lifecycle import Expiration
from boto.s3.lifecycle import Rule
from boto.s3.acl import Grant
from boto.s3.tagging import Tags, TagSet
from boto.s3.website import RedirectLocation
from boto.compat import unquote_str
def test_lifecycle_with_glacier_transition(self):
    lifecycle = Lifecycle()
    transition = Transition(days=30, storage_class='GLACIER')
    rule = Rule('myid', prefix='', status='Enabled', expiration=None, transition=transition)
    lifecycle.append(rule)
    self.assertTrue(self.bucket.configure_lifecycle(lifecycle))
    response = self.bucket.get_lifecycle_config()
    transition = response[0].transition
    self.assertEqual(transition.days, 30)
    self.assertEqual(transition.storage_class, 'GLACIER')
    self.assertEqual(transition.date, None)