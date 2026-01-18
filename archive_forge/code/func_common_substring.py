import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
def common_substring(s1, s2):
    """
    Returns the longest common substring to the two strings, starting from the
    left.
    """
    chunks = []
    path1 = splitall(s1)
    path2 = splitall(s2)
    for dir1, dir2 in zip(path1, path2):
        if dir1 != dir2:
            break
        chunks.append(dir1)
    return os.path.join(*chunks)