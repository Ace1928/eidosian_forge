import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def set_path_env_var(self, name):
    """Set environment variable 'name' to an MSVC path type value.

        This is equivalent to a SET command prior to execution of spawned
        commands.
        """
    if name == 'lib':
        p = self.get_msvc_paths('library')
    else:
        p = self.get_msvc_paths(name)
    if p:
        os.environ[name] = ';'.join(p)