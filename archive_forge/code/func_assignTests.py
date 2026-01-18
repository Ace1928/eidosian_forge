import json
import os
from os.path import dirname, abspath, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.kernel import SolverFactory, variable, maximize, minimize
from pyomo.solvers.tests.solvers import test_solver_cases as _test_solver_cases
def assignTests(cls, problem_list):
    for solver, writer in testing_solvers:
        for PROBLEM in problem_list:
            aux_list = [{'simplify': True}, {'simplify': False}]
            for AUX in aux_list:
                for REPN in ['sos2', 'mc', 'inc', 'cc', 'dcc', 'dlog', 'log']:
                    for BOUND_TYPE in ['lb', 'ub', 'eq']:
                        for SENSE in [maximize, minimize]:
                            if not (BOUND_TYPE == 'lb' and SENSE == maximize or (BOUND_TYPE == 'ub' and SENSE == minimize) or (REPN == 'mc' and 'step' in PROBLEM)):
                                kwds = {}
                                kwds['sense'] = SENSE
                                kwds['repn'] = REPN
                                kwds['bound'] = BOUND_TYPE
                                if SENSE == maximize:
                                    attrName = 'test_{0}_{1}_{2}_{3}_{4}_{5}'.format(PROBLEM, REPN, BOUND_TYPE, 'maximize', solver, writer)
                                else:
                                    assert SENSE == minimize
                                    attrName = 'test_{0}_{1}_{2}_{3}_{4}_{5}'.format(PROBLEM, REPN, BOUND_TYPE, 'minimize', solver, writer)
                                assert len(AUX) == 1
                                kwds.update(AUX)
                                attrName += '_simplify_' + str(AUX['simplify'])
                                setattr(cls, attrName, createTestMethod(attrName, PROBLEM, solver, writer, kwds))
                                with open(join(thisDir, 'kernel_baselines', PROBLEM + '_baseline_results.json'), 'r') as f:
                                    baseline_results = json.load(f)
                                    setattr(cls, PROBLEM + '_results', baseline_results)