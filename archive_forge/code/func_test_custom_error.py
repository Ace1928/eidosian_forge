from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_custom_error(self):

    class ZeroConstraint(object):

        def error(self, value):
            return '%s is not 0' % value

        def validate(self, value, context):
            return value == 0
    self.env.register_constraint('zero', ZeroConstraint)
    constraint = constraints.CustomConstraint('zero', environment=self.env)
    error = self.assertRaises(ValueError, constraint.validate, 1)
    self.assertEqual('1 is not 0', str(error))