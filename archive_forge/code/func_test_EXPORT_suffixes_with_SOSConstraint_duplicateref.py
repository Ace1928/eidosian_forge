import os
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
def test_EXPORT_suffixes_with_SOSConstraint_duplicateref(self):
    model = ConcreteModel()
    model.ref = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
    model.y = Var([1, 2, 3])
    model.obj = Objective(expr=sum_product(model.y))
    model.sos_con = SOSConstraint(var=model.y, index=[1, 2, 3], sos=1)
    for i, val in zip([1, 2, 3], [11, 12, 13]):
        model.ref.set_value(model.y[i], val)
    with self.assertRaisesRegex(RuntimeError, "NL file writer does not allow both manually declared 'ref' suffixes as well as SOSConstraint "):
        model.write(filename=os.path.join(self.tempdir, 'junk.nl'), format=self.nl_version, io_options={'symbolic_solver_labels': False})