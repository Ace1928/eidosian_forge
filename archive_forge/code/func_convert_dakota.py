import os
import sys
from pyomo.common.collections import Bunch
from pyomo.opt import ProblemFormat
from pyomo.core.base import Objective, Var, Constraint, value, ConcreteModel
def convert_dakota(options=Bunch(), parser=None):
    import pyomo.environ
    model_file = os.path.basename(options.model.save_file)
    model_file_no_ext = os.path.splitext(model_file)[0]
    if options.model.save_file is None:
        options.model.save_file = model_file_no_ext + '.nl'
    options.model.save_format = ProblemFormat.nl
    options.model.symbolic_solver_labels = True
    model_data = convert(options, parser)
    model = model_data.instance
    variables = 0
    var_descriptors = []
    var_lb = []
    var_ub = []
    var_initial = []
    tmpDict = model_data.symbol_map.getByObjectDictionary()
    for var in model.component_data_objects(Var, active=True):
        if id(var) in tmpDict:
            variables += 1
            var_descriptors.append(var.name)
            _lb, _ub = var.bounds
            if _lb is not None:
                var_lb.append(str(_lb))
            else:
                var_lb.append('-inf')
            if _ub is not None:
                var_ub.append(str(_ub))
            else:
                var_ub.append('inf')
            try:
                val = value(var)
            except:
                val = None
            var_initial.append(str(val))
    objectives = 0
    obj_descriptors = []
    for obj in model.component_data_objects(Objective, active=True):
        objectives += 1
        obj_descriptors.append(obj.name)
    constraints = 0
    cons_descriptors = []
    cons_lb = []
    cons_ub = []
    for con in model.component_data_objects(Constraint, active=True):
        constraints += 1
        cons_descriptors.append(con.name)
        if con.lower is not None:
            cons_lb.append(str(con.lower))
        else:
            cons_lb.append('-inf')
        if con.upper is not None:
            cons_ub.append(str(con.upper))
        else:
            cons_ub.append('inf')
    dakfrag = open(model_file_no_ext + '.dak', 'w')
    dakfrag.write('#--- Dakota variables block ---#\n')
    dakfrag.write('variables\n')
    dakfrag.write('  continuous_design ' + str(variables) + '\n')
    dakfrag.write('    descriptors\n')
    for vd in var_descriptors:
        dakfrag.write("      '%s'\n" % vd)
    dakfrag.write('    lower_bounds ' + ' '.join(var_lb) + '\n')
    dakfrag.write('    upper_bounds ' + ' '.join(var_ub) + '\n')
    dakfrag.write('    initial_point ' + ' '.join(var_initial) + '\n')
    dakfrag.write('#--- Dakota interface block ---#\n')
    dakfrag.write('interface\n')
    dakfrag.write("  algebraic_mappings = '" + options.model.save_file + "'\n")
    dakfrag.write('#--- Dakota responses block ---#\n')
    dakfrag.write('responses\n')
    dakfrag.write('  objective_functions ' + str(objectives) + '\n')
    if constraints > 0:
        dakfrag.write('  nonlinear_inequality_constraints ' + str(constraints) + '\n')
        dakfrag.write('    lower_bounds ' + ' '.join(cons_lb) + '\n')
        dakfrag.write('    upper_bounds ' + ' '.join(cons_ub) + '\n')
    dakfrag.write('    descriptors\n')
    for od in obj_descriptors:
        dakfrag.write("      '%s'\n" % od)
    if constraints > 0:
        for cd in cons_descriptors:
            dakfrag.write("      '%s'\n" % cd)
    dakfrag.write('  analytic_gradients\n')
    dakfrag.write('  no_hessians\n')
    dakfrag.close()
    sys.stdout.write("Dakota input fragment written to file '%s'\n" % (model_file_no_ext + '.dak',))
    return model_data