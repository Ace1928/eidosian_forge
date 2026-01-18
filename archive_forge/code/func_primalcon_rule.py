from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
def primalcon_rule(model, i):
    return sum((model.A[i, j] * model.x[j] for j in model.N)) >= model.b[i]