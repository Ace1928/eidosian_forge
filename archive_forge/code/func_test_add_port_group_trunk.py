import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch.object(dvs_util, 'get_port_group_spec')
def test_add_port_group_trunk(self, mock_spec):
    session = mock.Mock()
    dvs_moref = dvs_util.get_dvs_moref('dvs-123')
    spec = dvs_util.get_port_group_spec(session, 'pg', None, trunk_mode=True)
    mock_spec.return_value = spec
    dvs_util.add_port_group(session, dvs_moref, 'pg', trunk_mode=True)
    session.invoke_api.assert_called_once_with(session.vim, 'CreateDVPortgroup_Task', dvs_moref, spec=spec)