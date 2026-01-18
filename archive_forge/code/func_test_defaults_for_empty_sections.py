import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_defaults_for_empty_sections(self):
    """Test default secntion's content behavior of HOT template."""
    tmpl = template.Template(hot_tpl_empty_sections)
    self.assertIsInstance(tmpl, hot_template.HOTemplate20130523)
    self.assertNotIn('foobar', tmpl)
    self.assertEqual('No description', tmpl[tmpl.DESCRIPTION])
    self.assertEqual({}, tmpl[tmpl.RESOURCES])
    self.assertEqual({}, tmpl[tmpl.OUTPUTS])
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    self.assertIsNone(stack.parameters._validate_user_parameters())
    self.assertIsNone(stack.validate())