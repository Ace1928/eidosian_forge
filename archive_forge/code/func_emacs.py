import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def emacs(exe=u'emacs'):
    install_editor(exe + u' +{line} {filename}')