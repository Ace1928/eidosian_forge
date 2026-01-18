from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def input_dir(self, dirname):
    """Check all files in this directory and all subdirectories."""
    dirname = dirname.rstrip('/')
    if self.excluded(dirname):
        return 0
    counters = self.options.report.counters
    verbose = self.options.verbose
    filepatterns = self.options.filename
    runner = self.runner
    for root, dirs, files in os.walk(dirname):
        if verbose:
            print('directory ' + root)
        counters['directories'] += 1
        for subdir in sorted(dirs):
            if self.excluded(subdir, root):
                dirs.remove(subdir)
        for filename in sorted(files):
            if filename_match(filename, filepatterns) and (not self.excluded(filename, root)):
                runner(os.path.join(root, filename))