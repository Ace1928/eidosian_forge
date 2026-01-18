from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_create_db_instance_vpc_sg_str(self):
    self.set_http_response(status_code=200)
    vpc_security_groups = [VPCSecurityGroupMembership(self.service_connection, 'active', 'sg-1'), VPCSecurityGroupMembership(self.service_connection, None, 'sg-2')]
    db = self.service_connection.create_dbinstance('SimCoProd01', 10, 'db.m1.large', 'master', 'Password01', param_group='default.mysql5.1', db_subnet_group_name='dbSubnetgroup01', vpc_security_groups=vpc_security_groups)
    self.assert_request_parameters({'Action': 'CreateDBInstance', 'AllocatedStorage': 10, 'AutoMinorVersionUpgrade': 'true', 'DBInstanceClass': 'db.m1.large', 'DBInstanceIdentifier': 'SimCoProd01', 'DBParameterGroupName': 'default.mysql5.1', 'DBSubnetGroupName': 'dbSubnetgroup01', 'Engine': 'MySQL5.1', 'MasterUsername': 'master', 'MasterUserPassword': 'Password01', 'Port': 3306, 'VpcSecurityGroupIds.member.1': 'sg-1', 'VpcSecurityGroupIds.member.2': 'sg-2'}, ignore_params_values=['Version'])