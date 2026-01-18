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
def have_f90c(self):
    """Check for availability of Fortran 90 compiler.

        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.

        Notes
        -----
        True if a Fortran 90 compiler is available (because a simple Fortran
        90 code was able to be compiled successfully)
        """
    simple_fortran_subroutine = '\n        subroutine simple\n        end\n        '
    config_cmd = self.get_config_cmd()
    flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f90')
    return flag