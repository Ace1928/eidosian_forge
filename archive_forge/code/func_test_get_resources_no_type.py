import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_get_resources_no_type(self):
    """Test get resources with invalid key."""
    t = template_format.parse('\n            AWSTemplateFormatVersion: 2010-09-09\n            Resources:\n              resource1:\n                Properties:\n                  property1: value1\n                Metadata:\n                  foo: bar\n                DependsOn: dummy\n                DeletionPolicy: dummy\n                UpdatePolicy:\n                  foo: bar\n        ')
    tmpl = template.Template(t)
    error = self.assertRaises(exception.StackValidationFailed, tmpl.validate)
    self.assertEqual('Each Resource must contain a Type key.', str(error))