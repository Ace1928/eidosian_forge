import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def scite(exe=u'scite'):
    """ SciTE or Sc1 """
    install_editor(exe + u' {filename} -goto:{line}')