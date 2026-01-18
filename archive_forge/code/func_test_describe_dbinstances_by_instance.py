from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
def test_describe_dbinstances_by_instance(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_dbsnapshots(instance_id='simcoprod01')
    self.assert_request_parameters({'Action': 'DescribeDBSnapshots', 'DBInstanceIdentifier': 'simcoprod01'}, ignore_params_values=['Version'])
    self.assertEqual(len(response), 3)
    self.assertIsInstance(response[0], DBSnapshot)
    self.assertEqual(response[0].id, 'mydbsnapshot')
    self.assertEqual(response[0].status, 'available')
    self.assertEqual(response[0].instance_id, 'simcoprod01')
    self.assertEqual(response[0].engine_version, '5.1.50')
    self.assertEqual(response[0].license_model, 'general-public-license')
    self.assertEqual(response[0].iops, 1000)
    self.assertEqual(response[0].option_group_name, 'myoptiongroupname')
    self.assertEqual(response[0].percent_progress, 100)
    self.assertEqual(response[0].snapshot_type, 'manual')
    self.assertEqual(response[0].source_region, 'eu-west-1')
    self.assertEqual(response[0].vpc_id, 'myvpc')