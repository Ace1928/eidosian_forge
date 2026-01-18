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
def test_if_nullable_invalid_args(self):
    snippets = [{'if': ['create_prod']}, {'if': ['create_prod', 'one_value', 'two_values', 'three_values']}]
    tmpl = template.Template(hot_wallaby_tpl_empty)
    for snippet in snippets:
        exc = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
        self.assertIn('Arguments to "if" must be of the form: [condition_name, value_if_true, value_if_false]', str(exc))