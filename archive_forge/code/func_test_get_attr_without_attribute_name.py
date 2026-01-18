import copy
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions
from heat.engine import environment
from heat.engine import function
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_get_attr_without_attribute_name(self):
    ex = self.assertRaises(ValueError, functions.GetAtt, self.stack.defn, 'Fn::GetAtt', [self.rsrc.name])
    self.assertEqual('Arguments to "Fn::GetAtt" must be of the form [resource_name, attribute]', str(ex))