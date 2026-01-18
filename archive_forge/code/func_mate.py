import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def mate(exe=u'mate'):
    """ TextMate, the missing editor"""
    install_editor(exe + u' -w -l {line} {filename}')