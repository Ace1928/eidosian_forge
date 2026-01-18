import pyomo.contrib.piecewise.tests.models as models
from pyomo.core import Var
from pyomo.core.base import TransformationFactory
from pyomo.environ import value
from pyomo.gdp import Disjunct, Disjunction
def check_transformation_do_not_descend(test, transformation):
    m = models.make_log_x_model()
    transform = TransformationFactory(transformation)
    transform.apply_to(m)
    test.check_pw_log(m)
    test.check_pw_paraboloid(m)