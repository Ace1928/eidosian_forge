import os
import sys
from subprocess import PIPE, STDOUT, Popen
import pytest
import zmq
def run_mypy(*mypy_args):
    """Run mypy for a path

    Captures output and reports it on errors
    """
    p = Popen([sys.executable, '-m', 'mypy'] + list(mypy_args), stdout=PIPE, stderr=STDOUT)
    o, _ = p.communicate()
    out = o.decode('utf8', 'replace')
    print(out)
    assert p.returncode == 0, out