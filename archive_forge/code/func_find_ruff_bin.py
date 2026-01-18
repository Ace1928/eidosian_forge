import os
import sys
import sysconfig
def find_ruff_bin() -> str:
    """Return the ruff binary path."""
    ruff_exe = 'ruff' + sysconfig.get_config_var('EXE')
    path = os.path.join(sysconfig.get_path('scripts'), ruff_exe)
    if os.path.isfile(path):
        return path
    if sys.version_info >= (3, 10):
        user_scheme = sysconfig.get_preferred_scheme('user')
    elif os.name == 'nt':
        user_scheme = 'nt_user'
    elif sys.platform == 'darwin' and sys._framework:
        user_scheme = 'osx_framework_user'
    else:
        user_scheme = 'posix_user'
    path = os.path.join(sysconfig.get_path('scripts', scheme=user_scheme), ruff_exe)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)