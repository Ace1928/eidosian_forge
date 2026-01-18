from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
@ddt.data(('qos_policy', 'qos_object'), ('security_group', 'sg_object'), ('subnetpool', 'snp_object'), ('address_scope', 'as_object'), ('address_group', 'ag_object'))
@ddt.unpack
def test_network_rbac_create_object(self, obj_type, obj_fake_attr):
    obj_fake = getattr(self, obj_fake_attr)
    self.rbac_policy.object_type = obj_type
    self.rbac_policy.object_id = obj_fake.id
    arglist = ['--type', obj_type, '--action', self.rbac_policy.action, '--target-project', self.rbac_policy.target_project_id, obj_fake.name]
    verifylist = [('type', obj_type), ('action', self.rbac_policy.action), ('target_project', self.rbac_policy.target_project_id), ('rbac_object', obj_fake.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_rbac_policy.assert_called_with(**{'object_id': obj_fake.id, 'object_type': obj_type, 'action': self.rbac_policy.action, 'target_tenant': self.rbac_policy.target_project_id})
    self.data = [self.rbac_policy.action, self.rbac_policy.id, obj_fake.id, obj_type, self.rbac_policy.project_id, self.rbac_policy.target_project_id]
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))