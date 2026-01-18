import ast
import os
import platform
import re
import sys
from pyflakes import checker, __version__
from pyflakes import reporter as modReporter
def checkRecursive(paths, reporter):
    """
    Recursively check all source files in C{paths}.

    @param paths: A list of paths to Python source files and directories
        containing Python source files.
    @param reporter: A L{Reporter} where all of the warnings and errors
        will be reported to.
    @return: The number of warnings found.
    """
    warnings = 0
    for sourcePath in iterSourceCode(paths):
        warnings += checkPath(sourcePath, reporter)
    return warnings