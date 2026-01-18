from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_length_schema(self):
    d = {'length': {'min': 5, 'max': 10}, 'description': 'a length range'}
    r = constraints.Length(5, 10, description='a length range')
    self.assertEqual(d, dict(r))