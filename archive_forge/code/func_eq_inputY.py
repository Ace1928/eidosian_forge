from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.contrib.mindtpy.tests.MINLP_simple_grey_box import (
def eq_inputY(m):
    return m.Y[j] == m.my_block.egb.inputs['Y' + str(j)]