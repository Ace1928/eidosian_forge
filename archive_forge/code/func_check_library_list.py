import os
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler
from distutils import log
def check_library_list(self, libraries):
    """Ensure that the list of libraries is valid.

        `library` is presumably provided as a command option 'libraries'.
        This method checks that it is a list of 2-tuples, where the tuples
        are (library_name, build_info_dict).

        Raise DistutilsSetupError if the structure is invalid anywhere;
        just returns otherwise.
        """
    if not isinstance(libraries, list):
        raise DistutilsSetupError("'libraries' option must be a list of tuples")
    for lib in libraries:
        if not isinstance(lib, tuple) and len(lib) != 2:
            raise DistutilsSetupError("each element of 'libraries' must a 2-tuple")
        name, build_info = lib
        if not isinstance(name, str):
            raise DistutilsSetupError("first element of each tuple in 'libraries' must be a string (the library name)")
        if '/' in name or (os.sep != '/' and os.sep in name):
            raise DistutilsSetupError("bad library name '%s': may not contain directory separators" % lib[0])
        if not isinstance(build_info, dict):
            raise DistutilsSetupError("second element of each tuple in 'libraries' must be a dictionary (build info)")