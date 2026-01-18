import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
def get_class(name, notfound_action=1):
    """
    notfound_action:
      0 - do nothing
      1 - display warning message
      2 - raise error
    """
    cl = {'temp1': Temp1Info, 'temp2': Temp2Info, 'duplicate_options': DuplicateOptionInfo}.get(name.lower(), _system_info)
    return cl()