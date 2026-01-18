import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
def checkpath_iter(self, fullname):
    for dirpath in sys.path:
        finder = sys.path_importer_cache.get(dirpath)
        if isinstance(finder, (type(None), importlib.machinery.FileFinder)):
            checkpath = os.path.join(dirpath, '{0}.{1}'.format(fullname, self.suffix))
            yield checkpath