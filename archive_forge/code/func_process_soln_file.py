import os
import re
import time
import logging
import subprocess
from pyomo.common import Executable
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.kernel.block import IBlock
from pyomo.core import Var
from pyomo.opt.base import ProblemFormat, ResultsFormat, OptSolver
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
from pyomo.solvers.mockmip import MockMIP
def process_soln_file(self, results):
    extract_duals = False
    extract_reduced_costs = False
    for suffix in self._suffixes:
        flag = False
        if re.match(suffix, 'dual'):
            extract_duals = True
            flag = True
        if re.match(suffix, 'rc'):
            extract_reduced_costs = True
            flag = True
        if not flag:
            raise RuntimeError('***CBC solver plugin cannot extract solution suffix=' + suffix)
    if self._results_format is ResultsFormat.sol:
        return
    if len(results.solution) > 0:
        solution = results.solution(0)
    else:
        solution = Solution()
    results.problem.number_of_objectives = 1
    processing_constraints = None
    header_processed = False
    optim_value = None
    try:
        INPUT = open(self._soln_file, 'r')
    except IOError:
        INPUT = []
    _ver = self.version()
    invert_objective_sense = results.problem.sense == ProblemSense.maximize and (_ver and _ver[:3] < (2, 10, 2))
    for line in INPUT:
        tokens = tuple(re.split('[ \t]+', line.strip()))
        n_tokens = len(tokens)
        if not header_processed:
            if tokens[0] == 'Optimal':
                results.solver.termination_condition = TerminationCondition.optimal
                results.solver.status = SolverStatus.ok
                results.solver.termination_message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
                solution.status = SolutionStatus.optimal
                optim_value = _float(tokens[-1])
            elif tokens[0] in ('Infeasible', 'PrimalInfeasible') or (n_tokens > 1 and tokens[0:2] == ('Integer', 'infeasible')):
                results.solver.termination_message = 'Model was proven to be infeasible.'
                results.solver.termination_condition = TerminationCondition.infeasible
                results.solver.status = SolverStatus.warning
                solution.status = SolutionStatus.infeasible
                INPUT.close()
                return
            elif tokens[0] == 'Unbounded' or (n_tokens > 2 and tokens[0] == 'Problem' and (tokens[2] == 'unbounded')) or (n_tokens > 1 and tokens[0:2] == ('Dual', 'infeasible')):
                results.solver.termination_message = 'Model was proven to be unbounded.'
                results.solver.termination_condition = TerminationCondition.unbounded
                results.solver.status = SolverStatus.warning
                solution.status = SolutionStatus.unbounded
                INPUT.close()
                return
            elif n_tokens > 2 and tokens[0:2] == ('Stopped', 'on'):
                optim_value = _float(tokens[-1])
                solution.gap = None
                results.solver.status = SolverStatus.aborted
                solution.status = SolutionStatus.stoppedByLimit
                if tokens[2] == 'time':
                    results.solver.termination_message = 'Optimization terminated because the time expended exceeded the value specified in the seconds parameter.'
                    results.solver.termination_condition = TerminationCondition.maxTimeLimit
                elif tokens[2] == 'iterations':
                    if results.solver.termination_condition not in [TerminationCondition.maxEvaluations, TerminationCondition.other, TerminationCondition.maxIterations]:
                        results.solver.termination_message = 'Optimization terminated because a limit was hit'
                        results.solver.termination_condition = TerminationCondition.maxIterations
                elif tokens[2] == 'difficulties':
                    results.solver.termination_condition = TerminationCondition.solverFailure
                    results.solver.status = SolverStatus.error
                    solution.status = SolutionStatus.error
                elif tokens[2] == 'ctrl-c':
                    results.solver.termination_message = 'Optimization was terminated by the user.'
                    results.solver.termination_condition = TerminationCondition.userInterrupt
                    solution.status = SolutionStatus.unknown
                else:
                    results.solver.termination_condition = TerminationCondition.unknown
                    results.solver.status = SolverStatus.unknown
                    solution.status = SolutionStatus.unknown
                    results.solver.termination_message = ' '.join(tokens)
                    print('***WARNING: CBC plugin currently not processing solution status=Stopped correctly. Full status line is: {}'.format(line.strip()))
                if n_tokens > 8 and tokens[3:9] == ('(no', 'integer', 'solution', '-', 'continuous', 'used)'):
                    results.solver.termination_message = 'Optimization terminated because a limit was hit, however it had not found an integer solution yet.'
                    results.solver.termination_condition = TerminationCondition.intermediateNonInteger
                    solution.status = SolutionStatus.other
            else:
                results.solver.termination_condition = TerminationCondition.unknown
                results.solver.status = SolverStatus.unknown
                solution.status = SolutionStatus.unknown
                results.solver.termination_message = ' '.join(tokens)
                print('***WARNING: CBC plugin currently not processing solution status={} correctly. Full status line is: {}'.format(tokens[0], line.strip()))
        try:
            row_number = int(tokens[0])
            if row_number == 0:
                if processing_constraints is None:
                    processing_constraints = True
                elif processing_constraints is True:
                    processing_constraints = False
                else:
                    raise RuntimeError('CBC plugin encountered unexpected line=(' + line.strip() + ') in solution file=' + self._soln_file + '; constraint and variable sections already processed!')
        except ValueError:
            if tokens[0] in ('Optimal', 'Infeasible', 'Unbounded', 'Stopped', 'Integer', 'Status'):
                if optim_value is not None:
                    if invert_objective_sense:
                        optim_value *= -1
                    solution.objective['__default_objective__'] = {'Value': optim_value}
                header_processed = True
        if processing_constraints is True and extract_duals is True:
            if n_tokens == 4:
                pass
            elif n_tokens == 5 and tokens[0] == '**':
                tokens = tokens[1:]
            else:
                raise RuntimeError('Unexpected line format encountered in CBC solution file - line=' + line)
            constraint = tokens[1]
            constraint_ax = _float(tokens[2])
            constraint_dual = _float(tokens[3])
            if invert_objective_sense:
                constraint_dual *= -1
            if constraint[:2] == 'c_':
                solution.constraint[constraint] = {'Dual': constraint_dual}
            elif constraint[:2] == 'r_':
                existing_constraint_dual_dict = solution.constraint.get('r_l_' + constraint[4:], None)
                if existing_constraint_dual_dict:
                    existing_constraint_dual = existing_constraint_dual_dict['Dual']
                    if abs(constraint_dual) > abs(existing_constraint_dual):
                        solution.constraint['r_l_' + constraint[4:]] = {'Dual': constraint_dual}
                else:
                    solution.constraint['r_l_' + constraint[4:]] = {'Dual': constraint_dual}
        elif processing_constraints is False:
            if n_tokens == 4:
                pass
            elif n_tokens == 5 and tokens[0] == '**':
                tokens = tokens[1:]
            else:
                raise RuntimeError('Unexpected line format encountered in CBC solution file - line=' + line)
            variable_name = tokens[1]
            variable_value = _float(tokens[2])
            variable = solution.variable[variable_name] = {'Value': variable_value}
            if extract_reduced_costs is True:
                variable_reduced_cost = _float(tokens[3])
                if invert_objective_sense:
                    variable_reduced_cost *= -1
                variable['Rc'] = variable_reduced_cost
        elif header_processed is True:
            pass
        else:
            raise RuntimeError('CBC plugin encountered unexpected line=(' + line.strip() + ') in solution file=' + self._soln_file + '; expecting header, but found data!')
    if not type(INPUT) is list:
        INPUT.close()
    if len(results.solution) == 0 and solution.status in [SolutionStatus.optimal, SolutionStatus.stoppedByLimit, SolutionStatus.unknown, SolutionStatus.other]:
        results.solution.insert(solution)