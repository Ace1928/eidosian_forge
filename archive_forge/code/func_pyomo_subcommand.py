from pyomo.scripting.pyomo_parser import add_subparser, CustomHelpFormatter
from pyomo.common.deprecation import deprecated
def pyomo_subcommand(options):
    return install_extras(options.args, quiet=options.quiet)