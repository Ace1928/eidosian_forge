from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_get_all_logs_filtered(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_all_logs('db_instance_1', max_records=100, marker='error/mysql-error.log', file_size=2000000, filename_contains='error', file_last_written=12345678)
    self.assert_request_parameters({'Action': 'DescribeDBLogFiles', 'DBInstanceIdentifier': 'db_instance_1', 'MaxRecords': 100, 'Marker': 'error/mysql-error.log', 'FileSize': 2000000, 'FilenameContains': 'error', 'FileLastWritten': 12345678}, ignore_params_values=['Version'])
    self.assertEqual(len(response), 6)
    self.assertTrue(isinstance(response[0], LogFile))
    self.assertEqual(response[0].log_filename, 'error/mysql-error-running.log')
    self.assertEqual(response[0].last_written, '1364403600000')
    self.assertEqual(response[0].size, '0')