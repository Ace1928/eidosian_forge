from unittest import mock
import testtools
from heatclient.common import hook_utils
import heatclient.v1.shell as shell
def test_clear_wildcard_hooks(self):
    type(self.args).hook = mock.PropertyMock(return_value=['a/*b/bp*'])
    type(self.args).pre_create = mock.PropertyMock(return_value=True)
    a = mock.Mock()
    type(a).resource_name = 'a'
    b = mock.Mock()
    type(b).resource_name = 'matcthis_b'
    bp = mock.Mock()
    type(bp).resource_name = 'bp_matchthis'
    self.client.resources.list = mock.Mock(side_effect=[[a], [b], [bp]])
    m1 = mock.Mock()
    m2 = mock.Mock()
    type(m2).physical_resource_id = 'nested_id'
    self.client.resources.get = mock.Mock(side_effect=[m1, m2])
    shell.do_hook_clear(self.client, self.args)
    payload = self.client.resources.signal.call_args_list[0][1]
    self.assertEqual({'unset_hook': 'pre-create'}, payload['data'])
    self.assertEqual('bp_matchthis', payload['resource_name'])
    self.assertEqual('nested_id', payload['stack_id'])