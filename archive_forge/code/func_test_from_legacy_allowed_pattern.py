from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
def test_from_legacy_allowed_pattern(self):
    s = properties.Schema.from_legacy({'Type': 'String', 'AllowedPattern': '[a-z]*'})
    self.assertEqual(1, len(s.constraints))
    c = s.constraints[0]
    self.assertIsInstance(c, constraints.AllowedPattern)
    self.assertEqual('[a-z]*', c.pattern)