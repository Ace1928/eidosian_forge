import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
@staticmethod
def setup_detect_python2():
    """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
    if None in [RTs._rt_py2_detect, RTs._rtp_py2_detect]:
        RTs._rt_py2_detect = RefactoringTool(py2_detect_fixers)
        RTs._rtp_py2_detect = RefactoringTool(py2_detect_fixers, {'print_function': True})