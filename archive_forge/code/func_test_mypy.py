import os
import sys
from subprocess import PIPE, STDOUT, Popen
import pytest
import zmq
@pytest.mark.parametrize('filename', mypy_tests)
def test_mypy(filename):
    run_mypy('--disallow-untyped-calls', os.path.join(mypy_dir, filename))