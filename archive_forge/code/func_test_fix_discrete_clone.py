import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def test_fix_discrete_clone(self):
    self.model.A = RangeSet(1, 4)
    self.model.a = Var()
    self.model.b = Var(within=self.model.A)
    self.model.c = Var(within=NonNegativeIntegers)
    self.model.d = Var(within=Integers, bounds=(-2, 3))
    self.model.e = Var(within=Boolean)
    self.model.f = Var(domain=Boolean)
    instance = self.model.create_instance()
    instance_clone = instance.clone()
    xfrm = TransformationFactory('core.fix_discrete')
    rinst = xfrm.create_using(instance_clone)
    self.assertFalse(rinst.a.is_fixed())
    self.assertTrue(rinst.b.is_fixed())
    self.assertTrue(rinst.c.is_fixed())
    self.assertTrue(rinst.d.is_fixed())
    self.assertTrue(rinst.e.is_fixed())
    self.assertTrue(rinst.f.is_fixed())