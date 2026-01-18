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
def test_get_attr_all_attributes(self):
    tmpl = template.Template(self.hot_tpl)
    stack = parser.Stack(utils.dummy_context(), 'test_get_attr', tmpl)
    stack.store()
    if self.raises is None:
        dep_attrs = list(function.dep_attrs(tmpl.parse(stack.defn, self.snippet), 'resource1'))
    else:
        dep_attrs = []
    stack.create()
    self.assertEqual((parser.Stack.CREATE, parser.Stack.COMPLETE), stack.state)
    rsrc = stack['resource1']
    for action, status in ((rsrc.CREATE, rsrc.IN_PROGRESS), (rsrc.CREATE, rsrc.COMPLETE), (rsrc.RESUME, rsrc.IN_PROGRESS), (rsrc.RESUME, rsrc.COMPLETE), (rsrc.SUSPEND, rsrc.IN_PROGRESS), (rsrc.SUSPEND, rsrc.COMPLETE), (rsrc.UPDATE, rsrc.IN_PROGRESS), (rsrc.UPDATE, rsrc.COMPLETE), (rsrc.SNAPSHOT, rsrc.IN_PROGRESS), (rsrc.SNAPSHOT, rsrc.COMPLETE), (rsrc.CHECK, rsrc.IN_PROGRESS), (rsrc.CHECK, rsrc.COMPLETE), (rsrc.ADOPT, rsrc.IN_PROGRESS), (rsrc.ADOPT, rsrc.COMPLETE)):
        rsrc.state_set(action, status)
        with mock.patch.object(rsrc_defn.ResourceDefinition, 'dep_attrs') as mock_da:
            mock_da.return_value = dep_attrs
            node_data = rsrc.node_data()
        stk_defn.update_resource_data(stack.defn, rsrc.name, node_data)
        if self.raises is not None:
            ex = self.assertRaises(self.raises, self.resolve, self.snippet, tmpl, stack)
            self.assertEqual(self.expected, str(ex))
        else:
            self.assertEqual(self.expected, self.resolve(self.snippet, tmpl, stack))