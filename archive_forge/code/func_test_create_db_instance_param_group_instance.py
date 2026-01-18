from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_create_db_instance_param_group_instance(self):
    self.set_http_response(status_code=200)
    param_group = ParameterGroup()
    param_group.name = 'default.mysql5.1'
    db = self.service_connection.create_dbinstance('SimCoProd01', 10, 'db.m1.large', 'master', 'Password01', param_group=param_group, db_subnet_group_name='dbSubnetgroup01')
    self.assert_request_parameters({'Action': 'CreateDBInstance', 'AllocatedStorage': 10, 'AutoMinorVersionUpgrade': 'true', 'DBInstanceClass': 'db.m1.large', 'DBInstanceIdentifier': 'SimCoProd01', 'DBParameterGroupName': 'default.mysql5.1', 'DBSubnetGroupName': 'dbSubnetgroup01', 'Engine': 'MySQL5.1', 'MasterUsername': 'master', 'MasterUserPassword': 'Password01', 'Port': 3306}, ignore_params_values=['Version'])
    self.assertEqual(db.id, 'simcoprod01')
    self.assertEqual(db.engine, 'mysql')
    self.assertEqual(db.status, 'creating')
    self.assertEqual(db.allocated_storage, 10)
    self.assertEqual(db.instance_class, 'db.m1.large')
    self.assertEqual(db.master_username, 'master')
    self.assertEqual(db.multi_az, False)
    self.assertEqual(db.pending_modified_values, {'MasterUserPassword': '****'})
    self.assertEqual(db.parameter_group.name, 'default.mysql5.1')
    self.assertEqual(db.parameter_group.description, None)
    self.assertEqual(db.parameter_group.engine, None)