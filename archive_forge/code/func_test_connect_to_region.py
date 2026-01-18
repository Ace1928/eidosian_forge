from boto.exception import JSONResponseError
from boto.opsworks import connect_to_region, regions, RegionInfo
from boto.opsworks.layer1 import OpsWorksConnection
from tests.compat import unittest
def test_connect_to_region(self):
    connection = connect_to_region('us-east-1')
    self.assertIsInstance(connection, OpsWorksConnection)