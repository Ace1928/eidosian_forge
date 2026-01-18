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
@pytest.fixture(scope='session', autouse=True)
def sessionlevel_testing_path():
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ase-test-workdir-') as tempdir:
        path = Path(tempdir)
        path.chmod(365)
        with workdir(path):
            yield path
        path.chmod(493)