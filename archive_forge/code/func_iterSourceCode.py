import ast
import os
import platform
import re
import sys
from pyflakes import checker, __version__
from pyflakes import reporter as modReporter
def iterSourceCode(paths):
    """
    Iterate over all Python source files in C{paths}.

    @param paths: A list of paths.  Directories will be recursed into and
        any .py files found will be yielded.  Any non-directories will be
        yielded as-is.
    """
    for path in paths:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    full_path = os.path.join(dirpath, filename)
                    if isPythonFile(full_path):
                        yield full_path
        else:
            yield path