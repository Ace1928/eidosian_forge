from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_multi_args(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_log_file('db1', 'foo.log', marker='0:4485', number_of_lines=10)
    self.assertTrue(isinstance(response, LogFileObject))
    self.assert_request_parameters({'Action': 'DownloadDBLogFilePortion', 'DBInstanceIdentifier': 'db1', 'Marker': '0:4485', 'NumberOfLines': 10, 'LogFileName': 'foo.log'}, ignore_params_values=['Version'])