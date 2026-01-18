import pickle
import types
import pyomo.common.unittest as unittest
from pyomo.solvers.tests.models.base import all_models
from pyomo.solvers.tests.testcases import generate_scenarios
def pickle_test(self):
    model_class = test_case.model()
    model_class.generate_model(test_case.testcase.import_suffixes)
    model_class.warmstart_model()
    load_solutions = not model_class.solve_should_fail and test_case.status != 'expected failure'
    try:
        opt, status = model_class.solve(solver, io, test_case.testcase.io_options, test_case.testcase.options, symbolic_labels, load_solutions)
    except:
        if test_case.status == 'expected failure':
            return
        raise
    m = pickle.loads(pickle.dumps(model_class.model))
    instance1 = m.clone()
    model_class.model = instance1
    opt, status1 = model_class.solve(solver, io, test_case.testcase.io_options, test_case.testcase.options, symbolic_labels, load_solutions)
    inst, res = pickle.loads(pickle.dumps([instance1, status1]))
    instance2 = pickle.loads(pickle.dumps(instance1))
    self.assertNotEqual(id(instance1), id(instance2))
    model_class.model = instance2
    opt, status2 = model_class.solve(solver, io, test_case.testcase.io_options, test_case.testcase.options, symbolic_labels, load_solutions)
    inst, res = pickle.loads(pickle.dumps([instance2, status2]))
    instance3 = instance2.clone()
    self.assertNotEqual(id(instance2), id(instance3))
    model_class.model = instance3
    opt, status3 = model_class.solve(solver, io, test_case.testcase.io_options, test_case.testcase.options, symbolic_labels, load_solutions)
    inst, res = pickle.loads(pickle.dumps([instance3, status3]))