import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def get_package_dir(self, package):
    """Return the directory, relative to the top of the source
           distribution, where package 'package' should be found
           (at least according to the 'package_dir' option, if any)."""
    path = package.split('.')
    if not self.package_dir:
        if path:
            return os.path.join(*path)
        else:
            return ''
    else:
        tail = []
        while path:
            try:
                pdir = self.package_dir['.'.join(path)]
            except KeyError:
                tail.insert(0, path[-1])
                del path[-1]
            else:
                tail.insert(0, pdir)
                return os.path.join(*tail)
        else:
            pdir = self.package_dir.get('')
            if pdir is not None:
                tail.insert(0, pdir)
            if tail:
                return os.path.join(*tail)
            else:
                return ''