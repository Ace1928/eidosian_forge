import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def jed(exe=u'jed'):
    """ JED, the lightweight emacsish editor """
    install_editor(exe + u' +{line} {filename}')