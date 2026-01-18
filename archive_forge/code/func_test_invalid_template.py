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
def test_invalid_template(self):
    scanner_error = '\n            1\n            Mappings:\n              ValidMapping:\n                TestKey: TestValue\n            '
    parser_error = '\n            Mappings:\n              ValidMapping:\n                TestKey: {TestKey1: "Value1" TestKey2: "Value2"}\n            '
    self.assertRaises(ValueError, template_format.parse, scanner_error)
    self.assertRaises(ValueError, template_format.parse, parser_error)