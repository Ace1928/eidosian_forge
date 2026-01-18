import os
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
import pyomo.repn.plugins.ampl.ampl_ as ampl_
import pyomo.repn.plugins.nl_writer as nl_writer
def test_export_nonlinear_variables(self):
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.z = Var()
    model.w = Var([1, 2, 3])
    model.c = Constraint(expr=model.x == model.y ** 2)
    model.y.fix(3)
    test_fname = os.path.join(self.tempdir, 'export_nonlinear_variables')
    model.write(test_fname, format=self._nl_version, io_options={'symbolic_solver_labels': True})
    with open(test_fname + '.col') as f:
        names = list(map(str.strip, f.readlines()))
    assert 'z' not in names
    assert 'y' not in names
    assert 'x' in names
    model.write(test_fname, format=self._nl_version, io_options={'symbolic_solver_labels': True, 'export_nonlinear_variables': [model.z]})
    with open(test_fname + '.col') as f:
        names = list(map(str.strip, f.readlines()))
    assert 'z' in names
    assert 'y' not in names
    assert 'x' in names
    assert 'w[1]' not in names
    assert 'w[2]' not in names
    assert 'w[3]' not in names
    model.write(test_fname, format=self._nl_version, io_options={'symbolic_solver_labels': True, 'export_nonlinear_variables': [model.z, model.w]})
    with open(test_fname + '.col') as f:
        names = list(map(str.strip, f.readlines()))
    assert 'z' in names
    assert 'y' not in names
    assert 'x' in names
    assert 'w[1]' in names
    assert 'w[2]' in names
    assert 'w[3]' in names
    model.write(test_fname, format=self._nl_version, io_options={'symbolic_solver_labels': True, 'export_nonlinear_variables': [model.z, model.w[2]]})
    with open(test_fname + '.col') as f:
        names = list(map(str.strip, f.readlines()))
    assert 'z' in names
    assert 'y' not in names
    assert 'x' in names
    assert 'w[1]' not in names
    assert 'w[2]' in names
    assert 'w[3]' not in names