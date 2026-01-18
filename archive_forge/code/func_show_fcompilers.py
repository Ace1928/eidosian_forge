import os
import sys
import re
from pathlib import Path
from distutils.sysconfig import get_python_lib
from distutils.fancy_getopt import FancyGetopt
from distutils.errors import DistutilsModuleError, \
from distutils.util import split_quoted, strtobool
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils import log
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
from numpy.distutils.exec_command import find_executable
from numpy.distutils import _shell_utils
from .environment import EnvironmentConfig
def show_fcompilers(dist=None):
    """Print list of available compilers (used by the "--help-fcompiler"
    option to "config_fc").
    """
    if dist is None:
        from distutils.dist import Distribution
        from numpy.distutils.command.config_compiler import config_fc
        dist = Distribution()
        dist.script_name = os.path.basename(sys.argv[0])
        dist.script_args = ['config_fc'] + sys.argv[1:]
        try:
            dist.script_args.remove('--help-fcompiler')
        except ValueError:
            pass
        dist.cmdclass['config_fc'] = config_fc
        dist.parse_config_files()
        dist.parse_command_line()
    compilers = []
    compilers_na = []
    compilers_ni = []
    if not fcompiler_class:
        load_all_fcompiler_classes()
    platform_compilers = available_fcompilers_for_platform()
    for compiler in platform_compilers:
        v = None
        log.set_verbosity(-2)
        try:
            c = new_fcompiler(compiler=compiler, verbose=dist.verbose)
            c.customize(dist)
            v = c.get_version()
        except (DistutilsModuleError, CompilerNotFound) as e:
            log.debug('show_fcompilers: %s not found' % (compiler,))
            log.debug(repr(e))
        if v is None:
            compilers_na.append(('fcompiler=' + compiler, None, fcompiler_class[compiler][2]))
        else:
            c.dump_properties()
            compilers.append(('fcompiler=' + compiler, None, fcompiler_class[compiler][2] + ' (%s)' % v))
    compilers_ni = list(set(fcompiler_class.keys()) - set(platform_compilers))
    compilers_ni = [('fcompiler=' + fc, None, fcompiler_class[fc][2]) for fc in compilers_ni]
    compilers.sort()
    compilers_na.sort()
    compilers_ni.sort()
    pretty_printer = FancyGetopt(compilers)
    pretty_printer.print_help('Fortran compilers found:')
    pretty_printer = FancyGetopt(compilers_na)
    pretty_printer.print_help('Compilers available for this platform, but not found:')
    if compilers_ni:
        pretty_printer = FancyGetopt(compilers_ni)
        pretty_printer.print_help('Compilers not available on this platform:')
    print("For compiler details, run 'config_fc --verbose' setup command.")