import copy
import os
from pyomo.opt.base.formats import guess_format
from pyomo.opt.base.error import ConverterError
from pyomo.common import Factory
def convert_problem(args, target_problem_type, valid_problem_types, has_capability=lambda x: False, **kwds):
    """
    Convert a problem, defined by the 'args' tuple, into another
    problem.
    """
    if len(valid_problem_types) == 0:
        raise ConverterError('No valid problem types')
    if not (target_problem_type is None or target_problem_type in valid_problem_types):
        msg = "Problem type '%s' is not valid"
        raise ConverterError(msg % str(target_problem_type))
    if len(args) == 0:
        raise ConverterError('Empty argument list')
    tmp = args[0]
    if isinstance(tmp, str):
        fname = tmp.split(os.sep)[-1]
        if os.sep in fname:
            fname = tmp.split(os.sep)[-1]
        source_ptype = [guess_format(fname)]
        if source_ptype == [None]:
            raise ConverterError('Unknown suffix type: ' + tmp)
    else:
        source_ptype = args[0].valid_problem_types()
    valid_ptypes = copy.copy(valid_problem_types)
    if target_problem_type is not None:
        valid_ptypes.remove(target_problem_type)
        valid_ptypes = [target_problem_type] + valid_ptypes
    if source_ptype[0] in valid_ptypes:
        valid_ptypes.remove(source_ptype[0])
        valid_ptypes = [source_ptype[0]] + valid_ptypes
    for ptype in valid_ptypes:
        for s_ptype in source_ptype:
            if s_ptype == ptype:
                return (args, ptype, None)
            for name in ProblemConverterFactory:
                converter = ProblemConverterFactory(name)
                if converter.can_convert(s_ptype, ptype):
                    tmp = [s_ptype, ptype] + list(args)
                    tmp = tuple(tmp)
                    tmpkw = kwds
                    tmpkw['capabilities'] = has_capability
                    problem_files, symbol_map = converter.apply(*tmp, **tmpkw)
                    return (problem_files, ptype, symbol_map)
    msg = 'No conversion possible.  Source problem type: %s.  Valid target types: %s'
    raise ConverterError(msg % (str(source_ptype[0]), list(map(str, valid_ptypes))))