from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
def test_copy_dbinstance(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.copy_dbsnapshot('myautomaticdbsnapshot', 'mycopieddbsnapshot')
    self.assert_request_parameters({'Action': 'CopyDBSnapshot', 'SourceDBSnapshotIdentifier': 'myautomaticdbsnapshot', 'TargetDBSnapshotIdentifier': 'mycopieddbsnapshot'}, ignore_params_values=['Version'])
    self.assertIsInstance(response, DBSnapshot)
    self.assertEqual(response.id, 'mycopieddbsnapshot')
    self.assertEqual(response.status, 'available')