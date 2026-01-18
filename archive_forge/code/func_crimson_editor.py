import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def crimson_editor(exe=u'cedt.exe'):
    install_editor(exe + u' /L:{line} {filename}')