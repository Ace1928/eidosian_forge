from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_describe_option_group_options(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.describe_option_group_options()
    self.assertEqual(len(response), 1)
    options = response[0]
    self.assertEqual(options.name, 'OEM')
    self.assertEqual(options.description, 'Oracle Enterprise Manager')
    self.assertEqual(options.engine_name, 'oracle-se1')
    self.assertEqual(options.major_engine_version, '11.2')
    self.assertEqual(options.min_minor_engine_version, '0.2.v3')
    self.assertEqual(options.port_required, True)
    self.assertEqual(options.default_port, 1158)
    self.assertEqual(options.permanent, False)
    self.assertEqual(options.persistent, False)
    self.assertEqual(options.depends_on, [])