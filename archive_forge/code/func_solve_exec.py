import json
import sys
import argparse
from pyomo.opt import SolverFactory, UnknownSolver
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter
def solve_exec(args, unparsed):
    import pyomo.scripting.util
    solver_manager = getattr(args, 'solver_manager', None)
    solver = getattr(args, 'solver', None)
    if solver is None:
        try:
            val = pyomo.scripting.util.get_config_values(unparsed[-1])
        except IndexError:
            val = None
        except IOError:
            val = None
        if not val is None:
            try:
                solver = val['solvers'][0]['solver name']
            except:
                solver = None
        if solver is None:
            if not ('-h' in unparsed or '--help' in unparsed):
                print('ERROR: No solver specified!')
                print('')
            parser = create_temporary_parser(solver=True, generate=True)
            parser.parse_args(args=unparsed + ['-h'])
            sys.exit(1)
    config = None
    _solver = solver if solver_manager == 'serial' else 'asl'
    with SolverFactory(_solver) as opt:
        if opt is None or isinstance(opt, UnknownSolver):
            print("ERROR: Unknown solver '%s'!" % solver)
            sys.exit(1)
        if not args.template is None:
            config = opt.config_block(init=True)
            OUTPUT = open(args.template, 'w')
            if args.template.endswith('json'):
                OUTPUT.write(json.dumps(config.value(), indent=2))
            else:
                OUTPUT.write(config.generate_yaml_template())
            OUTPUT.close()
            print("  Created template file '%s'" % args.template)
            sys.exit(0)
        config = opt.config_block()
        if '-h' in unparsed or '--help' in unparsed:
            _parser = create_temporary_parser(generate=True)
        else:
            _parser = create_temporary_parser()
        config.initialize_argparse(_parser)
        _parser.usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
        _options = _parser.parse_args(args=unparsed)
        config.import_argparse(_options)
        config.solvers[0].solver_name = getattr(args, 'solver', None)
        if _options.model_or_config_file.endswith('.py'):
            config.model.filename = _options.model_or_config_file
            config.data.files = _options.data_files
        else:
            val = pyomo.scripting.util.get_config_values(_options.model_or_config_file)
            config.set_value(val)
    if config is None:
        raise RuntimeError('Failed to create config object')
    config.solvers[0].solver_name = solver
    config.solvers[0].manager = solver_manager
    from pyomo.scripting.pyomo_command import run_pyomo
    return pyomo.scripting.util.run_command(command=run_pyomo, parser=_parser, options=config, name='pyomo solve')