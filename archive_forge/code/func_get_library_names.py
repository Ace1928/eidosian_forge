import os
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler
from distutils import log
def get_library_names(self):
    if not self.libraries:
        return None
    lib_names = []
    for lib_name, build_info in self.libraries:
        lib_names.append(lib_name)
    return lib_names