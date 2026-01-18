import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def make_command(self, command=None):
    """Return command if one is passed, otherwise try to find
        ASE_VASP_COMMAND, VASP_COMMAND or VASP_SCRIPT.
        If none are set, a CalculatorSetupError is raised"""
    if command:
        cmd = command
    else:
        for env in self.env_commands:
            if env in os.environ:
                cmd = os.environ[env].replace('PREFIX', self.prefix)
                if env == 'VASP_SCRIPT':
                    exe = sys.executable
                    cmd = ' '.join([exe, cmd])
                break
        else:
            msg = 'Please set either command in calculator or one of the following environment variables (prioritized as follows): {}'.format(', '.join(self.env_commands))
            raise calculator.CalculatorSetupError(msg)
    return cmd