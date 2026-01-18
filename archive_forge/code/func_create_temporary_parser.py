import json
import sys
import argparse
from pyomo.opt import SolverFactory, UnknownSolver
from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter
def create_temporary_parser(solver=False, generate=False):
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    _subparsers = parser.add_subparsers()
    _parser = _subparsers.add_parser('solve')
    _parser.formatter_class = CustomHelpFormatter
    if generate:
        _parser.add_argument('--generate-config-template', action='store', dest='template', default=None, help='Create a configuration template file in YAML or JSON and exit.')
    if solver:
        _parser.add_argument('--solver', action='store', dest='solver', default=None, help='Solver name. This option is required unless the solver name isspecified in a configuration file.')
        _parser.usage = '%(prog)s [options] <model_or_config_file> [<data_files>]'
        _parser.epilog = "\nDescription:\n\nThe 'pyomo solve' subcommand optimizes a Pyomo model.  The --solver option\nis required because the specific steps executed are solver dependent.\nThe standard steps executed by this subcommand are:\n\n  - Apply pre-solve operations (e.g. import Python packages)\n  - Create the model\n  - Apply model transformations\n  - Perform optimization\n  - Apply post-solve operations (e.g. save optimal solutions)\n\n\nThis subcommand can be executed with or without a configuration file.\nThe command-line options can be used to perform simple optimization steps.\nFor example:\n\n  pyomo solve --solver=glpk model.py model.dat\n\nThis uses the 'glpk' solver to optimize the model define in 'model.py'\nusing the Pyomo data file 'pyomo.dat'.  Solver-specific command-line\noptions can be listed by specifying the '--solver' option and the '--help'\n(or '-h') option:\n\n  pyomo solve --solver=glpk --help\n\n\nA yaml or json configuration file can also be used to specify\noptions used by the solver.  For example:\n\n  pyomo solve --solver=glpk config.yaml\n\nNo other command-line options are required!  Further, the '--solver'\noption can be omitted if the solver name is specified in the configuration\nfile.\n\nConfiguration options are also solver-specific; a template configuration\nfile can be generated with the '--generate-config-template' option:\n\n  pyomo solve --solver=glpk --generate-config-template=template.yaml\n  pyomo solve --solver=glpk --generate-config-template=template.json\n\nNote that yaml template files contain comments that describe the\nconfiguration options.  Also, configuration files will generally support\nmore configuration options than are available with command-line options.\n\n"
    _parser.add_argument('model_or_config_file', action='store', nargs='?', default='', help="A Python module that defines a Pyomo model, or a configuration file that defines options for 'pyomo solve' (in either YAML or JSON format)")
    _parser.add_argument('data_files', action='store', nargs='*', default=[], help='Pyomo data files that defined data used to initialize the model (specified in the first argument)')
    return _parser