from unittest import mock
from openstack.compute.v2 import flavor as _flavor
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import flavor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_flavor_create_other_options(self):
    self.flavor.is_public = False
    arglist = ['--id', 'auto', '--ram', str(self.flavor.ram), '--disk', str(self.flavor.disk), '--ephemeral', str(self.flavor.ephemeral), '--swap', str(self.flavor.swap), '--vcpus', str(self.flavor.vcpus), '--rxtx-factor', str(self.flavor.rxtx_factor), '--private', '--description', str(self.flavor.description), '--project', self.project.id, '--property', 'key1=value1', '--property', 'key2=value2', self.flavor.name]
    verifylist = [('ram', self.flavor.ram), ('disk', self.flavor.disk), ('ephemeral', self.flavor.ephemeral), ('swap', self.flavor.swap), ('vcpus', self.flavor.vcpus), ('rxtx_factor', self.flavor.rxtx_factor), ('public', False), ('description', 'description'), ('project', self.project.id), ('properties', {'key1': 'value1', 'key2': 'value2'}), ('name', self.flavor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    args = {'name': self.flavor.name, 'ram': self.flavor.ram, 'vcpus': self.flavor.vcpus, 'disk': self.flavor.disk, 'id': 'auto', 'ephemeral': self.flavor.ephemeral, 'swap': self.flavor.swap, 'rxtx_factor': self.flavor.rxtx_factor, 'is_public': False, 'description': self.flavor.description}
    props = {'key1': 'value1', 'key2': 'value2'}
    create_flavor = _flavor.Flavor(**self.flavor)
    expected_flavor = _flavor.Flavor(**self.flavor)
    expected_flavor.extra_specs = props
    expected_flavor.is_public = False
    cmp_data = list(self.data_private)
    cmp_data[7] = format_columns.DictColumn(props)
    self.compute_sdk_client.create_flavor.return_value = create_flavor
    self.compute_sdk_client.create_flavor_extra_specs.return_value = expected_flavor
    with mock.patch.object(sdk_utils, 'supports_microversion', return_value=True):
        columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.create_flavor.assert_called_once_with(**args)
    self.compute_sdk_client.flavor_add_tenant_access.assert_called_with(self.flavor.id, self.project.id)
    self.compute_sdk_client.create_flavor_extra_specs.assert_called_with(create_flavor, props)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(cmp_data, data)