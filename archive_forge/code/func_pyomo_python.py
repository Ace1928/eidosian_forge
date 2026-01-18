import code
import sys
import subprocess
from pyomo.common._command import pyomo_command
from pyomo.common.deprecation import deprecated
import pyomo.scripting.pyomo_parser
@pyomo_command('pyomo_python', "Launch script using Pyomo's python installation")
@deprecated(msg="The 'pyomo_python' command has been deprecated and will be removed", version='6.0')
def pyomo_python(args=None):
    if args is None:
        args = sys.argv[1:]
    if args is None or len(args) == 0:
        console = code.InteractiveConsole()
        console.interact('Pyomo Python Console\n' + sys.version)
    else:
        cmd = sys.executable + ' ' + ' '.join(args)
        subprocess.run(cmd)