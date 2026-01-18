import os
import os.path
import sys
import glob
import textwrap
import logging
import socket
import subprocess
import pyomo.common
from pyomo.common.collections import Bunch
import pyomo.scripting.pyomo_parser
def help_solvers():
    import pyomo.environ
    wrapper = textwrap.TextWrapper(replace_whitespace=False)
    print('')
    print('Pyomo Solvers and Solver Managers')
    print('---------------------------------')
    print(wrapper.fill("Pyomo uses 'solver managers' to execute 'solvers' that perform optimization and other forms of model analysis.  A solver directly executes an optimizer, typically using an executable found on the user's PATH environment.  Solver managers support a flexible mechanism for asynchronously executing solvers either locally or remotely.  The following solver managers are available in Pyomo:"))
    print('')
    solvermgr_list = list(pyomo.opt.SolverManagerFactory)
    solvermgr_list = sorted(filter(lambda x: '_' != x[0], solvermgr_list))
    n = max(map(len, solvermgr_list))
    wrapper = textwrap.TextWrapper(subsequent_indent=' ' * (n + 9))
    for s in solvermgr_list:
        format = '    %-' + str(n) + 's     %s'
        print(wrapper.fill(format % (s, pyomo.opt.SolverManagerFactory.doc(s))))
    print('')
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(wrapper.fill('If no solver manager is specified, Pyomo uses the serial solver manager to execute solvers locally.  The neos solver manager is used to execute solvers on the NEOS optimization server.'))
    print('')
    print('')
    print('Serial Solver Interfaces')
    print('------------------------')
    print(wrapper.fill('The serial manager supports the following solver interfaces:'))
    print('')
    solver_list = list(pyomo.opt.SolverFactory)
    solver_list = sorted(filter(lambda x: '_' != x[0], solver_list))
    _data = []
    try:
        logging.disable(logging.WARNING)
        for s in solver_list:
            with pyomo.opt.SolverFactory(s) as opt:
                ver = ''
                if opt.available(False):
                    avail = '-'
                    if opt.license_is_valid():
                        avail = '+'
                    try:
                        ver = opt.version()
                        if ver:
                            while len(ver) > 2 and ver[-1] == 0:
                                ver = ver[:-1]
                            ver = '.'.join((str(v) for v in ver))
                        else:
                            ver = ''
                    except (AttributeError, NameError):
                        pass
                elif s == 'py' or (hasattr(opt, '_metasolver') and opt._metasolver):
                    avail = '*'
                else:
                    avail = ''
                _data.append((avail, s, ver, pyomo.opt.SolverFactory.doc(s)))
    finally:
        logging.disable(logging.NOTSET)
    nameFieldLen = max((len(line[1]) for line in _data))
    verFieldLen = max((len(line[2]) for line in _data))
    fmt = '   %%1s%%-%ds %%-%ds %%s' % (nameFieldLen, verFieldLen)
    wrapper = textwrap.TextWrapper(subsequent_indent=' ' * (nameFieldLen + verFieldLen + 6))
    for _line in _data:
        print(wrapper.fill(fmt % _line))
    print('')
    wrapper = textwrap.TextWrapper(subsequent_indent='')
    print(wrapper.fill('The leading symbol (one of *, -, +) indicates the current solver availability.  A plus (+) indicates the solver is currently available to be run from Pyomo with the serial solver manager, and (if applicable) has a valid license.  A minus (-) indicates the solver executables are available but do not report having a valid license.  The solver may still be usable in an unlicensed or "demo" mode for limited problem sizes. An asterisk (*) indicates meta-solvers or generic interfaces, which are always available.'))
    print('')
    print(wrapper.fill('Pyomo also supports solver interfaces that are wrappers around third-party solver interfaces. These interfaces require a subsolver specification that indicates the solver being executed.  For example, the following indicates that the ipopt solver will be used:'))
    print('')
    print('   asl:ipopt')
    print('')
    print(wrapper.fill('The asl interface provides a generic wrapper for all solvers that use the AMPL Solver Library.'))
    print('')
    print(wrapper.fill("Note that subsolvers can not be enumerated automatically for these interfaces.  However, if a solver is specified that is not found, Pyomo assumes that the asl solver interface is being used.  Thus the following solver name will launch ipopt if the 'ipopt' executable is on the user's path:"))
    print('')
    print('   ipopt')
    print('')
    try:
        logging.disable(logging.WARNING)
        socket.setdefaulttimeout(10)
        import pyomo.neos.kestrel
        kestrel = pyomo.neos.kestrel.kestrelAMPL()
        solver_list = list(set([name[:-5].lower() for name in kestrel.solvers() if name.endswith('AMPL')]))
        if len(solver_list) > 0:
            print('')
            print('NEOS Solver Interfaces')
            print('----------------------')
            print(wrapper.fill('The neos solver manager supports solver interfaces that can be executed remotely on the NEOS optimization server.  The following solver interfaces are available with your current system configuration:'))
            print('')
            solver_list = sorted(solver_list)
            n = max(map(len, solver_list))
            format = '    %-' + str(n) + 's     %s'
            for name in solver_list:
                print(wrapper.fill(format % (name, pyomo.neos.doc.get(name, 'Unexpected NEOS solver'))))
            print('')
        else:
            print('')
            print('NEOS Solver Interfaces')
            print('----------------------')
            print(wrapper.fill('The neos solver manager supports solver interfaces that can be executed remotely on the NEOS optimization server.  This server is not available with your current system configuration.'))
            print('')
    except ImportError:
        pass
    finally:
        logging.disable(logging.NOTSET)
        socket.setdefaulttimeout(None)