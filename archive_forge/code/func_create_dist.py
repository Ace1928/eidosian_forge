import os
import sys
import shutil
import tempfile
import unittest
import sysconfig
from copy import deepcopy
from test.support import os_helper
from distutils import log
from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL
from distutils.core import Distribution
def create_dist(self, pkg_name='foo', **kw):
    """Will generate a test environment.

        This function creates:
         - a Distribution instance using keywords
         - a temporary directory with a package structure

        It returns the package directory and the distribution
        instance.
        """
    tmp_dir = self.mkdtemp()
    pkg_dir = os.path.join(tmp_dir, pkg_name)
    os.mkdir(pkg_dir)
    dist = Distribution(attrs=kw)
    return (pkg_dir, dist)