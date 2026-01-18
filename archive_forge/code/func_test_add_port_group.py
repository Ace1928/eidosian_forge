import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch.object(dvs_util, 'get_port_group_spec')
def test_add_port_group(self, mock_spec):
    session = mock.Mock()
    dvs_moref = dvs_util.get_dvs_moref('dvs-123')
    spec = dvs_util.get_port_group_spec(session, 'pg', 7)
    mock_spec.return_value = spec
    pg_moref = vim_util.get_moref('dvportgroup-7', 'DistributedVirtualPortgroup')

    def wait_for_task_side_effect(task):
        task_info = mock.Mock()
        task_info.result = pg_moref
        return task_info
    session.wait_for_task.side_effect = wait_for_task_side_effect
    pg = dvs_util.add_port_group(session, dvs_moref, 'pg', vlan_id=7)
    self.assertEqual(pg, pg_moref)
    session.invoke_api.assert_called_once_with(session.vim, 'CreateDVPortgroup_Task', dvs_moref, spec=spec)