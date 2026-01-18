from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
def test_delete_dbinstance(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.delete_dbsnapshot('mysnapshot2')
    self.assert_request_parameters({'Action': 'DeleteDBSnapshot', 'DBSnapshotIdentifier': 'mysnapshot2'}, ignore_params_values=['Version'])
    self.assertIsInstance(response, DBSnapshot)
    self.assertEqual(response.id, 'mysnapshot2')
    self.assertEqual(response.status, 'deleted')