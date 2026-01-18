import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
from numpy.distutils.ccompiler_opt import new_ccompiler_opt
def get_source_files(self):
    self.check_library_list(self.libraries)
    filenames = []
    for lib in self.libraries:
        filenames.extend(get_lib_source_files(lib))
    return filenames