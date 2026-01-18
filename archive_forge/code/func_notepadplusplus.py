import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def notepadplusplus(exe=u'notepad++'):
    """ Notepad++ http://notepad-plus.sourceforge.net """
    install_editor(exe + u' -n{line} {filename}')