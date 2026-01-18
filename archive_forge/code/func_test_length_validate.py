from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_length_validate(self):
    cl = constraints.Length(min=5, max=5, description='a range')
    cl.validate('abcde')