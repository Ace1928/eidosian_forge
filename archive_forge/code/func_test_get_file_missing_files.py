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
def test_get_file_missing_files(self):
    """Test get_file function with no matching key in files section."""
    snippet = {'get_file': 'file:///tmp/foo.yaml'}
    tmpl = template.Template(hot_tpl_empty, files={'file:///tmp/bar.yaml': 'bar contents'})
    stack = parser.Stack(utils.dummy_context(), 'param_id_test', tmpl)
    missingErr = self.assertRaises(ValueError, self.resolve, snippet, tmpl, stack)
    self.assertEqual('No content found in the "files" section for get_file path: file:///tmp/foo.yaml', str(missingErr))