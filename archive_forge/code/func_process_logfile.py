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
def process_logfile(self):
    """
        Process logfile
        """
    results = SolverResults()
    if self._problem_format is ProblemFormat.nl:
        return results
    soln = Solution()
    OUTPUT = open(self._log_file)
    output = ''.join(OUTPUT.readlines())
    OUTPUT.close()
    results.problem.sense = ProblemSense.minimize
    results.problem.name = None
    optim_value = float('inf')
    lower_bound = None
    upper_bound = None
    gap = None
    nodes = None
    for line in output.split('\n'):
        tokens = tuple(re.split('[ \t]+', line.strip()))
        n_tokens = len(tokens)
        if n_tokens > 1:
            if n_tokens > 4 and tokens[:4] == ('Continuous', 'objective', 'value', 'is'):
                lower_bound = _float(tokens[4])
            elif n_tokens > 12 and tokens[1:3] == ('Search', 'completed') and (tokens[4:6] == ('best', 'objective')) and (tokens[9] == 'iterations') and (tokens[12] == 'nodes'):
                optim_value = _float(tokens[6][:-1])
                results.solver.statistics.black_box.number_of_iterations = int(tokens[8])
                nodes = int(tokens[11])
            elif tokens[1] == 'Exiting' and n_tokens > 4:
                if tokens[2:4] == ('on', 'maximum'):
                    results.solver.termination_condition = {'nodes': TerminationCondition.maxEvaluations, 'time': TerminationCondition.maxTimeLimit, 'solutions': TerminationCondition.other, 'iterations': TerminationCondition.maxIterations}.get(tokens[4], TerminationCondition.other)
            elif n_tokens >= 4 and tokens[1:4] == ('Integer', 'solution', 'of'):
                optim_value = _float(tokens[4])
                try:
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[tokens.index('iterations') - 1])
                    nodes = int(tokens[tokens.index('nodes') - 1])
                except ValueError:
                    pass
            elif n_tokens > 15 and tokens[1:3] == ('Partial', 'search') and (tokens[4:6] == ('best', 'objective')) and (tokens[7:9] == ('(best', 'possible')) and (tokens[12] == 'iterations') and (tokens[15] == 'nodes'):
                optim_value = _float(tokens[6])
                lower_bound = _float(tokens[9][:-2])
                results.solver.statistics.black_box.number_of_iterations = int(tokens[11])
                nodes = int(tokens[14])
            elif n_tokens > 12 and tokens[1] == 'After' and (tokens[3] == 'nodes,') and (tokens[8:10] == ('best', 'solution,')) and (tokens[10:12] == ('best', 'possible')):
                nodes = int(tokens[2])
                optim_value = _float(tokens[7])
                lower_bound = _float(tokens[12])
            elif tokens[0] == 'Current' and n_tokens == 10 and (tokens[1] == 'default') and (tokens[2] == '(if') and (results.problem.name is None):
                results.problem.name = tokens[-1]
                if '.' in results.problem.name:
                    parts = results.problem.name.split('.')
                    if len(parts) > 2:
                        results.problem.name = '.'.join(parts[:-1])
                    else:
                        results.problem.name = results.problem.name.split('.')[0]
                if '/' in results.problem.name:
                    results.problem.name = results.problem.name.split('/')[-1]
                if '\\' in results.problem.name:
                    results.problem.name = results.problem.name.split('\\')[-1]
            elif tokens[0] == 'Presolve':
                if n_tokens > 9 and tokens[3] == 'rows,' and (tokens[6] == 'columns'):
                    results.problem.number_of_variables = int(tokens[4]) - int(tokens[5][1:-1])
                    results.problem.number_of_constraints = int(tokens[1]) - int(tokens[2][1:-1])
                    results.problem.number_of_objectives = 1
                elif n_tokens > 6 and tokens[6] == 'infeasible':
                    soln.status = SolutionStatus.infeasible
            elif n_tokens > 11 and tokens[:2] == ('Problem', 'has') and (tokens[3] == 'rows,') and (tokens[5] == 'columns') and (tokens[7:9] == ('with', 'objective)')):
                results.problem.number_of_variables = int(tokens[4])
                results.problem.number_of_constraints = int(tokens[2])
                results.problem.number_of_nonzeros = int(tokens[6][1:])
                results.problem.number_of_objectives = 1
            elif n_tokens > 8 and tokens[:3] == ('Original', 'problem', 'has') and (tokens[4] == 'integers') and (tokens[6:9] == ('of', 'which', 'binary)')):
                results.problem.number_of_integer_variables = int(tokens[3])
                results.problem.number_of_binary_variables = int(tokens[5][1:])
            elif n_tokens == 5 and tokens[3] == 'NAME':
                results.problem.name = tokens[4]
            elif 'CoinLpIO::readLp(): Maximization problem reformulated as minimization' in ' '.join(tokens):
                results.problem.sense = ProblemSense.maximize
            elif n_tokens > 3 and tokens[:2] == ('Result', '-'):
                if tokens[2:4] in [('Run', 'abandoned'), ('User', 'ctrl-c')]:
                    results.solver.termination_condition = TerminationCondition.userInterrupt
                if n_tokens > 4:
                    if tokens[2:5] == ('Optimal', 'solution', 'found'):
                        soln.status = SolutionStatus.optimal
                    elif tokens[2:5] in [('Linear', 'relaxation', 'infeasible'), ('Problem', 'proven', 'infeasible')]:
                        soln.status = SolutionStatus.infeasible
                    elif tokens[2:5] == ('Linear', 'relaxation', 'unbounded'):
                        soln.status = SolutionStatus.unbounded
                    elif n_tokens > 5 and tokens[2:4] == ('Stopped', 'on') and (tokens[5] == 'limit'):
                        results.solver.termination_condition = {'node': TerminationCondition.maxEvaluations, 'time': TerminationCondition.maxTimeLimit, 'solution': TerminationCondition.other, 'iterations': TerminationCondition.maxIterations}.get(tokens[4], TerminationCondition.other)
                elif n_tokens > 3 and tokens[2] == 'Finished':
                    soln.status = SolutionStatus.optimal
                    optim_value = _float(tokens[4])
            elif n_tokens >= 3 and tokens[:2] == ('Objective', 'value:'):
                optim_value = _float(tokens[2])
            elif n_tokens >= 4 and tokens[:4] == ('No', 'feasible', 'solution', 'found'):
                soln.status = SolutionStatus.infeasible
            elif n_tokens > 2 and tokens[:2] == ('Lower', 'bound:'):
                if lower_bound is None:
                    results.problem.lower_bound = _float(tokens[2])
            elif tokens[0] == 'Gap:':
                gap = _float(tokens[1])
            elif n_tokens > 2 and tokens[:2] == ('Enumerated', 'nodes:'):
                nodes = int(tokens[2])
            elif n_tokens > 2 and tokens[:2] == ('Total', 'iterations:'):
                results.solver.statistics.black_box.number_of_iterations = int(tokens[2])
            elif n_tokens > 3 and tokens[:3] == ('Time', '(CPU', 'seconds):'):
                results.solver.system_time = _float(tokens[3])
            elif n_tokens > 3 and tokens[:3] == ('Time', '(Wallclock', 'Seconds):'):
                results.solver.wallclock_time = _float(tokens[3])
            elif n_tokens > 4 and tokens[:4] == ('Total', 'time', '(CPU', 'seconds):'):
                results.solver.system_time = _float(tokens[4])
                if n_tokens > 7 and tokens[5:7] == ('(Wallclock', 'seconds):'):
                    results.solver.wallclock_time = _float(tokens[7])
            elif tokens[0] == 'Optimal':
                if n_tokens > 4 and tokens[2] == 'objective' and (tokens[4] != 'and'):
                    soln.status = SolutionStatus.optimal
                    optim_value = _float(tokens[4])
                elif n_tokens > 5 and tokens[1] == 'objective' and (tokens[5] == 'iterations'):
                    soln.status = SolutionStatus.optimal
                    optim_value = _float(tokens[2])
                    results.solver.statistics.black_box.number_of_iterations = int(tokens[4])
            elif tokens[0] == 'sys' and n_tokens == 2:
                results.solver.system_time = _float(tokens[1])
            elif tokens[0] == 'user' and n_tokens == 2:
                results.solver.user_time = _float(tokens[1])
            elif n_tokens == 10 and 'Presolve' in tokens and ('iterations' in tokens) and (tokens[0] == 'Optimal') and ('objective' == tokens[1]):
                soln.status = SolutionStatus.optimal
                optim_value = _float(tokens[2])
            results.solver.user_time = -1.0
    if results.problem.name is None:
        results.problem.name = 'unknown'
    if soln.status is SolutionStatus.optimal:
        results.solver.termination_message = 'Model was solved to optimality (subject to tolerances), and an optimal solution is available.'
        results.solver.termination_condition = TerminationCondition.optimal
        results.solver.status = SolverStatus.ok
        if gap is None:
            lower_bound = optim_value
            gap = 0.0
    elif soln.status == SolutionStatus.infeasible:
        results.solver.termination_message = 'Model was proven to be infeasible.'
        results.solver.termination_condition = TerminationCondition.infeasible
        results.solver.status = SolverStatus.warning
    elif soln.status == SolutionStatus.unbounded:
        results.solver.termination_message = 'Model was proven to be unbounded.'
        results.solver.termination_condition = TerminationCondition.unbounded
        results.solver.status = SolverStatus.warning
    elif results.solver.termination_condition in [TerminationCondition.maxTimeLimit, TerminationCondition.maxEvaluations, TerminationCondition.other, TerminationCondition.maxIterations]:
        results.solver.status = SolverStatus.aborted
        soln.status = SolutionStatus.stoppedByLimit
        if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            results.solver.termination_message = 'Optimization terminated because the time expended exceeded the value specified in the seconds parameter.'
        elif results.solver.termination_condition == TerminationCondition.maxEvaluations:
            results.solver.termination_message = 'Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the maxNodes parameter'
        elif results.solver.termination_condition == TerminationCondition.other:
            results.solver.termination_message = 'Optimization terminated because the number of solutions found reached the value specified in the maxSolutions parameter.'
        elif results.solver.termination_condition == TerminationCondition.maxIterations:
            results.solver.termination_message = 'Optimization terminated because the total number of simplex iterations performed exceeded the value specified in the maxIterations parameter.'
    soln.gap = gap
    if results.problem.sense == ProblemSense.minimize:
        upper_bound = optim_value
    elif results.problem.sense == ProblemSense.maximize:
        _ver = self.version()
        if _ver and _ver[:3] < (2, 10, 2):
            optim_value *= -1
            upper_bound = None if lower_bound is None else -lower_bound
        else:
            upper_bound = None if lower_bound is None else lower_bound
        lower_bound = optim_value
    soln.objective['__default_objective__'] = {'Value': optim_value}
    results.problem.lower_bound = lower_bound
    results.problem.upper_bound = upper_bound
    results.solver.statistics.branch_and_bound.number_of_bounded_subproblems = nodes
    results.solver.statistics.branch_and_bound.number_of_created_subproblems = nodes
    if soln.status in [SolutionStatus.optimal, SolutionStatus.stoppedByLimit, SolutionStatus.unknown, SolutionStatus.other]:
        results.solution.insert(soln)
    return results