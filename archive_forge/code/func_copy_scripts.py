import os, re
from stat import ST_MODE
from distutils import sysconfig
from distutils.core import Command
from distutils.dep_util import newer
from distutils.util import convert_path, Mixin2to3
from distutils import log
import tokenize
def copy_scripts(self):
    outfiles, updated_files = build_scripts.copy_scripts(self)
    if not self.dry_run:
        self.run_2to3(updated_files)
    return (outfiles, updated_files)