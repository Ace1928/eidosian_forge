from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_allowed_pattern_schema(self):
    d = {'allowed_pattern': '[A-Za-z0-9]', 'description': 'alphanumeric'}
    r = constraints.AllowedPattern('[A-Za-z0-9]', description='alphanumeric')
    self.assertEqual(d, dict(r))