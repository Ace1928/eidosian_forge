from typing import Tuple, Dict, Any, List
import io
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
def parse_sol_file(sol_file: io.TextIOBase, nl_info: NLWriterInfo, result: Results) -> Tuple[Results, SolFileData]:
    sol_data = SolFileData()
    message = []
    for line in sol_file:
        if not line:
            break
        line = line.strip()
        if 'Options' in line:
            break
        message.append(line)
    message = '\n'.join(message)
    model_objects = []
    if 'Options' in line:
        line = sol_file.readline()
        number_of_options = int(line)
        if number_of_options > 4:
            raise DeveloperError('\nThe sol file reader has hit an unexpected error while parsing. The number of\noptions recorded is greater than 4. Please report this error to the Pyomo\ndevelopers.\n                                 ')
        for i in range(number_of_options + 4):
            line = sol_file.readline()
            model_objects.append(int(line))
    else:
        raise PyomoException("ERROR READING `sol` FILE. No 'Options' line found.")
    number_of_cons = model_objects[number_of_options + 1]
    number_of_vars = model_objects[number_of_options + 3]
    assert number_of_cons == len(nl_info.constraints)
    assert number_of_vars == len(nl_info.variables)
    duals = [float(sol_file.readline()) for i in range(number_of_cons)]
    variable_vals = [float(sol_file.readline()) for i in range(number_of_vars)]
    exit_code = [0, 0]
    line = sol_file.readline()
    if line and 'objno' in line:
        exit_code_line = line.split()
        if len(exit_code_line) != 3:
            raise PyomoException(f'ERROR READING `sol` FILE. Expected two numbers in `objno` line; received {line}.')
        exit_code = [int(exit_code_line[1]), int(exit_code_line[2])]
    else:
        raise PyomoException(f'ERROR READING `sol` FILE. Expected `objno`; received {line}.')
    result.extra_info.solver_message = message.strip().replace('\n', '; ')
    exit_code_message = ''
    if exit_code[1] >= 0 and exit_code[1] <= 99:
        result.solution_status = SolutionStatus.optimal
        result.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
    elif exit_code[1] >= 100 and exit_code[1] <= 199:
        exit_code_message = 'Optimal solution indicated, but ERROR LIKELY!'
        result.solution_status = SolutionStatus.feasible
        result.termination_condition = TerminationCondition.error
    elif exit_code[1] >= 200 and exit_code[1] <= 299:
        exit_code_message = 'INFEASIBLE SOLUTION: constraints cannot be satisfied!'
        result.solution_status = SolutionStatus.infeasible
        result.termination_condition = TerminationCondition.locallyInfeasible
    elif exit_code[1] >= 300 and exit_code[1] <= 399:
        exit_code_message = 'UNBOUNDED PROBLEM: the objective can be improved without limit!'
        result.solution_status = SolutionStatus.noSolution
        result.termination_condition = TerminationCondition.unbounded
    elif exit_code[1] >= 400 and exit_code[1] <= 499:
        exit_code_message = 'EXCEEDED MAXIMUM NUMBER OF ITERATIONS: the solver was stopped by a limit that you set!'
        result.solution_status = SolutionStatus.infeasible
        result.termination_condition = TerminationCondition.iterationLimit
    elif exit_code[1] >= 500 and exit_code[1] <= 599:
        exit_code_message = 'FAILURE: the solver stopped by an error condition in the solver routines!'
        result.termination_condition = TerminationCondition.error
    if result.extra_info.solver_message:
        if exit_code_message:
            result.extra_info.solver_message += '; ' + exit_code_message
    else:
        result.extra_info.solver_message = exit_code_message
    if result.solution_status != SolutionStatus.noSolution:
        sol_data.primals = variable_vals
        sol_data.duals = duals
        line = sol_file.readline()
        while line:
            line = line.strip()
            if line == '':
                continue
            line = line.split()
            if line[0] != 'suffix':
                remaining = ''
                line = sol_file.readline()
                while line:
                    remaining += line.strip() + '; '
                    line = sol_file.readline()
                result.extra_info.solver_message += remaining
                break
            read_data_type = int(line[1])
            data_type = read_data_type & 3
            convert_function = int
            if read_data_type & 4 == 4:
                convert_function = float
            number_of_entries = int(line[2])
            number_of_string_lines = int(line[5])
            suffix_name = sol_file.readline().strip()
            for line in range(number_of_string_lines):
                sol_data.other.append(sol_file.readline())
            if data_type == 0:
                sol_data.var_suffixes[suffix_name] = dict()
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    var_ndx = int(suf_line[0])
                    sol_data.var_suffixes[suffix_name][var_ndx] = convert_function(suf_line[1])
            elif data_type == 1:
                sol_data.con_suffixes[suffix_name] = dict()
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    con_ndx = int(suf_line[0])
                    sol_data.con_suffixes[suffix_name][con_ndx] = convert_function(suf_line[1])
            elif data_type == 2:
                sol_data.obj_suffixes[suffix_name] = dict()
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    obj_ndx = int(suf_line[0])
                    sol_data.obj_suffixes[suffix_name][obj_ndx] = convert_function(suf_line[1])
            elif data_type == 3:
                sol_data.problem_suffixes[suffix_name] = list()
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    sol_data.problem_suffixes[suffix_name].append(convert_function(suf_line[1]))
            line = sol_file.readline()
    return (result, sol_data)