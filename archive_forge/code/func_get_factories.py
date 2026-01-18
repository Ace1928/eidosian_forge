import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def get_factories(pytestconfig):
    opt = pytestconfig.getoption('--calculators')
    requested_calculators = opt.split(',') if opt else []
    return Factories(requested_calculators)