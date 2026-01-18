from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
def test_8PP_deactivate(self):
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    m = exfile.build_eight_process_flowsheet()
    for djn in m.component_data_objects(ctype=Disjunction):
        djn.deactivate()
    self.assertTrue(satisfiable(m) is not False)