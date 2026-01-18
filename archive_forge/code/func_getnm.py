import re
import sys
import subprocess
def getnm(nm_cmd=['nm', '-Cs', 'python%s.lib' % py_ver], shell=True):
    """Returns the output of nm_cmd via a pipe.

nm_output = getnm(nm_cmd = 'nm -Cs py_lib')"""
    p = subprocess.Popen(nm_cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    nm_output, nm_err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('failed to run "%s": "%s"' % (' '.join(nm_cmd), nm_err))
    return nm_output