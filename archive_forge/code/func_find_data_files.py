import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def find_data_files(self, package, src_dir):
    """Return filenames for package's data files in 'src_dir'"""
    globs = self.package_data.get('', []) + self.package_data.get(package, [])
    files = []
    for pattern in globs:
        filelist = glob.glob(os.path.join(glob.escape(src_dir), convert_path(pattern)))
        files.extend([fn for fn in filelist if fn not in files and os.path.isfile(fn)])
    return files