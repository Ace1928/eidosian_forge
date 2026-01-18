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
def test_filter_wrong_args_number(self):
    snippet = {'filter': [[None], [1, 2], 'foo']}
    tmpl = template.Template(hot_ocata_tpl_empty)
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl)
    self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl, stack=stack)