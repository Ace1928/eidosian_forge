from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_allowed_values_numeric_int(self):
    """Test AllowedValues constraint for numeric integer values.

        Test if the AllowedValues constraint works for numeric values in any
        combination of numeric strings or numbers in the constraint and
        numeric strings or numbers as value.
        """
    schema = constraints.Schema('Integer', constraints=[constraints.AllowedValues([1, 2, 4])])
    self.assertIsNone(schema.validate_constraints(1))
    err = self.assertRaises(exception.StackValidationFailed, schema.validate_constraints, 3)
    self.assertEqual('3 is not an allowed value [1, 2, 4]', str(err))
    self.assertIsNone(schema.validate_constraints('1'))
    err = self.assertRaises(exception.StackValidationFailed, schema.validate_constraints, '3')
    self.assertEqual('"3" is not an allowed value [1, 2, 4]', str(err))
    schema = constraints.Schema('Integer', constraints=[constraints.AllowedValues(['1', '2', '4'])])
    self.assertIsNone(schema.validate_constraints(1))
    err = self.assertRaises(exception.StackValidationFailed, schema.validate_constraints, 3)
    self.assertEqual('3 is not an allowed value ["1", "2", "4"]', str(err))
    self.assertIsNone(schema.validate_constraints('1'))
    err = self.assertRaises(exception.StackValidationFailed, schema.validate_constraints, '3')
    self.assertEqual('"3" is not an allowed value ["1", "2", "4"]', str(err))