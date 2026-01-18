import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor
def run_raw(command):
    """
    Runs the string command, returns any output.
    """
    proc = subprocess.Popen(command, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, env=_make_env())
    data = proc.stdout.read()
    proc.wait()
    while data.endswith('\n') or data.endswith('\r'):
        data = data[:-1]
    if data:
        data = '\n'.join([l for l in data.splitlines() if l])
        return data
    else:
        return ''