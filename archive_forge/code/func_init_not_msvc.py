from ctypes.util import find_library
from subprocess import check_output, CalledProcessError, DEVNULL
import ctypes
import os
import sys
import sysconfig
def init_not_msvc(self):
    """ Find OpenMP library and try to load if using ctype interface. """
    env_vars = []
    if sys.platform == 'darwin':
        env_vars = ['DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']
    else:
        env_vars = ['LD_LIBRARY_PATH']
    paths = []
    for env_var in env_vars:
        env_paths = os.environ.get(env_var, '')
        if env_paths:
            paths.extend(env_paths.split(os.pathsep))
    libomp_names = self.get_libomp_names()
    if cxx is not None:
        for libomp_name in libomp_names:
            cmd = [cxx, '-print-file-name=lib{}{}'.format(libomp_name, get_shared_lib_extension())]
            try:
                output = check_output(cmd, stderr=DEVNULL)
                path = os.path.dirname(output.decode().strip())
                if path:
                    paths.append(path)
            except (OSError, CalledProcessError):
                pass
    for libomp_name in libomp_names:
        libomp_path = find_library(libomp_name)
        if not libomp_path:
            for path in paths:
                candidate_path = os.path.join(path, 'lib{}{}'.format(libomp_name, get_shared_lib_extension()))
                if os.path.isfile(candidate_path):
                    libomp_path = candidate_path
                    break
        if libomp_path:
            try:
                self.libomp = ctypes.CDLL(libomp_path)
            except OSError:
                raise ImportError("found openMP library '{}' but couldn't load it. This may happen if you are cross-compiling.".format(libomp_path))
            self.version = 45
            return
    raise ImportError("I can't find a shared library for libomp, you may need to install it or adjust the {} environment variable.".format(env_vars[0]))