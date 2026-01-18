from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_no_description(self):
    p = new_parameter('p', {'Type': self.p_type}, validate_value=False)
    self.assertEqual('', p.description())