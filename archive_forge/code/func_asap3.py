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
@pytest.fixture(scope='session')
def asap3():
    return pytest.importorskip('asap3')