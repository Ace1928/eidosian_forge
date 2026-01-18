import sys
import json, os, sysconfig
def links_against_libpython():
    if sys.version_info >= (3, 8) and (not is_pypy):
        variables = sysconfig.get_config_vars()
        return bool(variables.get('LIBPYTHON', 'yes'))
    else:
        from distutils.core import Distribution, Extension
        cmd = Distribution().get_command_obj('build_ext')
        cmd.ensure_finalized()
        return bool(cmd.get_libraries(Extension('dummy', [])))