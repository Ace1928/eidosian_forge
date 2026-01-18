from unittest import mock
from heat.engine.clients.os import nova
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_aggregate_handle_update_hosts(self):
    ag = mock.MagicMock()
    ag.hosts = ['host_1', 'host_2']
    self.aggregates.get.return_value = ag
    prop_diff = {'hosts': ['host_1', 'host_3']}
    add_host_expected = 'host_3'
    remove_host_expected = 'host_2'
    self.my_aggregate.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.assertEqual(0, ag.update.call_count)
    self.assertEqual(0, ag.set_metadata.call_count)
    ag.add_host.assert_called_once_with(add_host_expected)
    ag.remove_host.assert_called_once_with(remove_host_expected)