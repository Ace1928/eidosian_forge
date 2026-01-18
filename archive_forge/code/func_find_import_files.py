from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def find_import_files(self):
    """ return a list of files to import """
    if self.files_to_tests:
        pyfiles = self.files_to_tests.keys()
    else:
        pyfiles = []
        for base_dir in self.files_or_dirs:
            if os.path.isdir(base_dir):
                for root, dirs, files in os.walk(base_dir):
                    exclude = {}
                    for d in dirs:
                        for init in ['__init__.py', '__init__.pyo', '__init__.pyc', '__init__.pyw', '__init__$py.class']:
                            if os.path.exists(os.path.join(root, d, init).replace('\\', '/')):
                                break
                        else:
                            exclude[d] = 1
                    if exclude:
                        new = []
                        for d in dirs:
                            if d not in exclude:
                                new.append(d)
                        dirs[:] = new
                    self.__add_files(pyfiles, root, files)
            elif os.path.isfile(base_dir):
                pyfiles.append(base_dir)
    if self.configuration.exclude_files or self.configuration.include_files:
        ret = []
        for f in pyfiles:
            add = True
            basename = os.path.basename(f)
            if self.configuration.include_files:
                add = False
                for pat in self.configuration.include_files:
                    if fnmatch.fnmatchcase(basename, pat):
                        add = True
                        break
            if not add:
                if self.verbosity > 3:
                    sys.stdout.write('Skipped file: %s (did not match any include_files pattern: %s)\n' % (f, self.configuration.include_files))
            elif self.configuration.exclude_files:
                for pat in self.configuration.exclude_files:
                    if fnmatch.fnmatchcase(basename, pat):
                        if self.verbosity > 3:
                            sys.stdout.write('Skipped file: %s (matched exclude_files pattern: %s)\n' % (f, pat))
                        elif self.verbosity > 2:
                            sys.stdout.write('Skipped file: %s\n' % (f,))
                        add = False
                        break
            if add:
                if self.verbosity > 3:
                    sys.stdout.write('Adding file: %s for test discovery.\n' % (f,))
                ret.append(f)
        pyfiles = ret
    return pyfiles