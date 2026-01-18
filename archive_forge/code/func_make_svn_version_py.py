import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def make_svn_version_py(self, delete=True):
    """Appends a data function to the data_files list that will generate
        __svn_version__.py file to the current package directory.

        Generate package __svn_version__.py file from SVN revision number,
        it will be removed after python exits but will be available
        when sdist, etc commands are executed.

        Notes
        -----
        If __svn_version__.py existed before, nothing is done.

        This is
        intended for working with source directories that are in an SVN
        repository.
        """
    target = njoin(self.local_path, '__svn_version__.py')
    revision = self._get_svn_revision(self.local_path)
    if os.path.isfile(target) or revision is None:
        return
    else:

        def generate_svn_version_py():
            if not os.path.isfile(target):
                version = str(revision)
                self.info('Creating %s (version=%r)' % (target, version))
                with open(target, 'w') as f:
                    f.write('version = %r\n' % version)

            def rm_file(f=target, p=self.info):
                if delete:
                    try:
                        os.remove(f)
                        p('removed ' + f)
                    except OSError:
                        pass
                    try:
                        os.remove(f + 'c')
                        p('removed ' + f + 'c')
                    except OSError:
                        pass
            atexit.register(rm_file)
            return target
        self.add_data_files(('', generate_svn_version_py()))