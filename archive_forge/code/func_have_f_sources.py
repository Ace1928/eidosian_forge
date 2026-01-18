import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
from numpy.distutils.ccompiler_opt import new_ccompiler_opt
def have_f_sources(self):
    for lib_name, build_info in self.libraries:
        if has_f_sources(build_info.get('sources', [])):
            return True
    return False