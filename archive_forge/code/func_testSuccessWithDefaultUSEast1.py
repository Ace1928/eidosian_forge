from tests.compat import unittest
from boto.s3.connection import S3Connection
from boto.s3 import connect_to_region
def testSuccessWithDefaultUSEast1(self):
    connection = connect_to_region('us-east-1')
    self.assertEquals('s3.amazonaws.com', connection.host)
    self.assertIsInstance(connection, S3Connection)