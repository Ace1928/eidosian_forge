from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
def test_do_post_op_failure(self):
    lcp_mappings = [('A::B::C1', TestLifecycleCallout1), ('A::B::C5', TestLifecycleCallout5)]
    self.mock_lcp_class_map(lcp_mappings)
    mc = mock.Mock()
    mc.__setattr__('pre_counter_for_unit_test', 0)
    mc.__setattr__('post_counter_for_unit_test', 0)
    ms = mock.Mock()
    ms.__setattr__('action', 'A')
    lifecycle_plugin_utils.do_post_ops(mc, ms, None, None)
    self.assertEqual(1, mc.post_counter_for_unit_test)
    self.mock_get_plugins.assert_called_once_with()