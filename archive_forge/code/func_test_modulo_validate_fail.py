from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_modulo_validate_fail(self):
    r = constraints.Modulo(step=2, offset=1)
    err = self.assertRaises(ValueError, r.validate, 4)
    self.assertIn('4 is not a multiple of 2 with an offset of 1', str(err))
    self.assertRaises(ValueError, r.validate, 0)
    self.assertRaises(ValueError, r.validate, 2)
    self.assertRaises(ValueError, r.validate, 888888)
    r = constraints.Modulo(step=2, offset=0)
    self.assertRaises(ValueError, r.validate, 1)
    self.assertRaises(ValueError, r.validate, 3)
    self.assertRaises(ValueError, r.validate, 5)
    self.assertRaises(ValueError, r.validate, 777777)
    err = self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, step=111, offset=111)
    self.assertIn('offset must be smaller (by absolute value) than step', str(err))
    err = self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, step=111, offset=112)
    self.assertIn('offset must be smaller (by absolute value) than step', str(err))
    err = self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, step=0, offset=1)
    self.assertIn('step cannot be 0', str(err))
    err = self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, step=-2, offset=1)
    self.assertIn('step and offset must be both positive or both negative', str(err))
    err = self.assertRaises(exception.InvalidSchemaError, constraints.Modulo, step=2, offset=-1)
    self.assertIn('step and offset must be both positive or both negative', str(err))