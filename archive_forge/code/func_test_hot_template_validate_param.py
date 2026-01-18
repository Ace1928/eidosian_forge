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
def test_hot_template_validate_param(self):
    len_desc = 'string length should be between 8 and 16'
    pattern_desc1 = 'Value must consist of characters only'
    pattern_desc2 = 'Value must start with a lowercase character'
    hot_tpl = template_format.parse('\n        heat_template_version: 2013-05-23\n        parameters:\n          db_name:\n            description: The WordPress database name\n            type: string\n            default: wordpress\n            constraints:\n              - length: { min: 8, max: 16 }\n                description: %s\n              - allowed_pattern: "[a-zA-Z]+"\n                description: %s\n              - allowed_pattern: "[a-z]+[a-zA-Z]*"\n                description: %s\n        ' % (len_desc, pattern_desc1, pattern_desc2))
    tmpl = template.Template(hot_tpl)

    def run_parameters(value):
        tmpl.parameters(identifier.HeatIdentifier('', 'stack_testit', None), {'db_name': value}).validate(validate_value=True)
        return True
    value = 'wp'
    err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
    self.assertIn(len_desc, str(err))
    value = 'abcdefghijklmnopq'
    err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
    self.assertIn(len_desc, str(err))
    value = 'abcdefgh1'
    err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
    self.assertIn(pattern_desc1, str(err))
    value = 'Abcdefghi'
    err = self.assertRaises(exception.StackValidationFailed, run_parameters, value)
    self.assertIn(pattern_desc2, str(err))
    value = 'abcdefghi'
    self.assertTrue(run_parameters(value))
    value = 'abcdefghI'
    self.assertTrue(run_parameters(value))