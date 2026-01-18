import os
import sys
from subprocess import Popen
import importlib
from pathlib import Path
import warnings
import argparse
from multiprocessing import cpu_count
from ase.calculators.calculator import names as calc_names
from ase.cli.main import CLIError
def all_test_modules():
    for abspath in testdir.rglob('test_*.py'):
        path = abspath.relative_to(testdir)
        yield path