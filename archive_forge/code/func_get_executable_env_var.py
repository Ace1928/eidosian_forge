import os
import shutil
import importlib
from ase.calculators.calculator import names
def get_executable_env_var(name):
    return 'ASE_{}_COMMAND'.format(name.upper())