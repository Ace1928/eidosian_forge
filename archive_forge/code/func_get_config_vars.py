import os
import sys
from os.path import pardir, realpath
def get_config_vars(*args):
    """With no arguments, return a dictionary of all configuration
    variables relevant for the current platform.

    On Unix, this means every variable defined in Python's installed Makefile;
    On Windows it's a much smaller set.

    With arguments, return a list of values that result from looking up
    each argument in the configuration variable dictionary.
    """
    global _CONFIG_VARS
    if _CONFIG_VARS is None:
        _CONFIG_VARS = {}
        _CONFIG_VARS['prefix'] = _PREFIX
        _CONFIG_VARS['exec_prefix'] = _EXEC_PREFIX
        _CONFIG_VARS['py_version'] = _PY_VERSION
        _CONFIG_VARS['py_version_short'] = _PY_VERSION_SHORT
        _CONFIG_VARS['py_version_nodot'] = _PY_VERSION_SHORT_NO_DOT
        _CONFIG_VARS['installed_base'] = _BASE_PREFIX
        _CONFIG_VARS['base'] = _PREFIX
        _CONFIG_VARS['installed_platbase'] = _BASE_EXEC_PREFIX
        _CONFIG_VARS['platbase'] = _EXEC_PREFIX
        _CONFIG_VARS['projectbase'] = _PROJECT_BASE
        _CONFIG_VARS['platlibdir'] = sys.platlibdir
        try:
            _CONFIG_VARS['abiflags'] = sys.abiflags
        except AttributeError:
            _CONFIG_VARS['abiflags'] = ''
        try:
            _CONFIG_VARS['py_version_nodot_plat'] = sys.winver.replace('.', '')
        except AttributeError:
            _CONFIG_VARS['py_version_nodot_plat'] = ''
        if os.name == 'nt':
            _init_non_posix(_CONFIG_VARS)
            _CONFIG_VARS['VPATH'] = sys._vpath
        if os.name == 'posix':
            _init_posix(_CONFIG_VARS)
        if _HAS_USER_BASE:
            _CONFIG_VARS['userbase'] = _getuserbase()
        srcdir = _CONFIG_VARS.get('srcdir', _PROJECT_BASE)
        if os.name == 'posix':
            if _PYTHON_BUILD:
                base = os.path.dirname(get_makefile_filename())
                srcdir = os.path.join(base, srcdir)
            else:
                srcdir = os.path.dirname(get_makefile_filename())
        _CONFIG_VARS['srcdir'] = _safe_realpath(srcdir)
        if sys.platform == 'darwin':
            import _osx_support
            _osx_support.customize_config_vars(_CONFIG_VARS)
    if args:
        vals = []
        for name in args:
            vals.append(_CONFIG_VARS.get(name))
        return vals
    else:
        return _CONFIG_VARS