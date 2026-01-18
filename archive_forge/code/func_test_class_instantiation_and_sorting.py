from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
def test_class_instantiation_and_sorting(self):
    lcp_mappings = []
    self.mock_lcp_class_map(lcp_mappings)
    pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
    self.assertEqual(0, len(pp_cis))
    self.mock_get_plugins.assert_called_once_with()
    lcp_mappings = [('A::B::C2', TestLifecycleCallout2), ('A::B::C1', TestLifecycleCallout1)]
    self.mock_lcp_class_map(lcp_mappings)
    pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
    self.assertEqual(2, len(pp_cis))
    self.assertEqual(100, pp_cis[0].get_ordinal())
    self.assertEqual(101, pp_cis[1].get_ordinal())
    self.assertEqual(TestLifecycleCallout1, pp_cis[0].__class__)
    self.assertEqual(TestLifecycleCallout2, pp_cis[1].__class__)
    self.mock_get_plugins.assert_called_once_with()
    lcp_mappings = [('A::B::C1', TestLifecycleCallout1), ('A::B::C2', TestLifecycleCallout2)]
    self.mock_lcp_class_map(lcp_mappings)
    pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
    self.assertEqual(2, len(pp_cis))
    self.assertEqual(100, pp_cis[0].get_ordinal())
    self.assertEqual(101, pp_cis[1].get_ordinal())
    self.assertEqual(TestLifecycleCallout1, pp_cis[0].__class__)
    self.assertEqual(TestLifecycleCallout2, pp_cis[1].__class__)
    self.mock_get_plugins.assert_called_once_with()
    lcp_mappings = [('A::B::C2', TestLifecycleCallout2), ('A::B::C3', TestLifecycleCallout3), ('A::B::C1', TestLifecycleCallout1)]
    self.mock_lcp_class_map(lcp_mappings)
    pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
    self.assertEqual(3, len(pp_cis))
    self.assertEqual(100, pp_cis[2].get_ordinal())
    self.assertEqual(101, pp_cis[0].get_ordinal())
    self.assertEqual(TestLifecycleCallout2, pp_cis[0].__class__)
    self.assertEqual(TestLifecycleCallout3, pp_cis[1].__class__)
    self.assertEqual(TestLifecycleCallout1, pp_cis[2].__class__)
    self.mock_get_plugins.assert_called_once_with()