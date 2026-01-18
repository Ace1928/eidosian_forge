import sys
import os
from pathlib import Path
from subprocess import Popen, PIPE, check_output
import zlib
import pytest
import numpy as np
import ase
from ase.utils import workdir, seterr
from ase.test.factories import (CalculatorInputs,
from ase.dependencies import all_dependencies
def ase(self, *args, expect_fail=False):
    environment = {}
    environment.update(os.environ)
    environment['MPLBACKEND'] = 'Agg'
    proc = Popen(['ase', '-T'] + list(args), stdout=PIPE, stdin=PIPE, env=environment)
    stdout, _ = proc.communicate(b'')
    status = proc.wait()
    assert (status != 0) == expect_fail
    return stdout.decode('utf-8')