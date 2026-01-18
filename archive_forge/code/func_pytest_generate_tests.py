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
def pytest_generate_tests(metafunc):
    from ase.test.factories import parametrize_calculator_tests
    parametrize_calculator_tests(metafunc)
    if 'seed' in metafunc.fixturenames:
        seeds = metafunc.config.getoption('seed')
        if len(seeds) == 0:
            seeds = [0]
        else:
            seeds = list(map(int, seeds))
        metafunc.parametrize('seed', seeds)