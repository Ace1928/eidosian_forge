import os
import os.path as op
import sys
from ase.io import read
from ase.io.formats import filetype, UnknownFileTypeError
from ase.db import connect
from ase.db.core import parse_selection
from ase.db.jsondb import JSONDatabase
from ase.db.row import atoms2dict
def allpaths(folder, include, exclude):
    """Generate paths."""
    exclude += ['.py', '.pyc']
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in filenames:
            if any((name.endswith(ext) for ext in exclude)):
                continue
            if include:
                for ext in include:
                    if name.endswith(ext):
                        break
                else:
                    continue
            path = op.join(dirpath, name)
            yield path
        dirnames[:] = (name for name in dirnames if name[0] not in '._')