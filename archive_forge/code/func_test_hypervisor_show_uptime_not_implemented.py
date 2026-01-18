import json
from unittest import mock
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.compute.v2 import hypervisor
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
@mock.patch.object(sdk_utils, 'supports_microversion', side_effect=[False, True, False])
def test_hypervisor_show_uptime_not_implemented(self, sm_mock):
    arglist = [self.hypervisor.name]
    verifylist = [('hypervisor', self.hypervisor.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.compute_sdk_client.get_hypervisor_uptime.side_effect = nova_exceptions.HTTPNotImplemented(501)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ('aggregates', 'cpu_info', 'current_workload', 'disk_available_least', 'free_disk_gb', 'free_ram_mb', 'host_ip', 'hypervisor_hostname', 'hypervisor_type', 'hypervisor_version', 'id', 'local_gb', 'local_gb_used', 'memory_mb', 'memory_mb_used', 'running_vms', 'service_host', 'service_id', 'state', 'status', 'vcpus', 'vcpus_used')
    expected_data = ([], format_columns.DictColumn({'aaa': 'aaa'}), 0, 50, 50, 1024, '192.168.0.10', self.hypervisor.name, 'QEMU', 2004001, self.hypervisor.id, 50, 0, 1024, 512, 0, 'aaa', 1, 'up', 'enabled', 4, 0)
    self.assertEqual(expected_columns, columns)
    self.assertCountEqual(expected_data, data)