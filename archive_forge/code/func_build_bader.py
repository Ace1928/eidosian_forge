from __future__ import annotations
import os
import shutil
import subprocess
from glob import glob
from typing import TYPE_CHECKING, Literal
from urllib.request import urlretrieve
from monty.json import jsanitize
from monty.serialization import dumpfn, loadfn
from ruamel import yaml
from pymatgen.core import OLD_SETTINGS_FILE, SETTINGS_FILE, Element
from pymatgen.io.cp2k.inputs import GaussianTypeOrbitalBasisSet, GthPotential
from pymatgen.io.cp2k.utils import chunk
def build_bader(fortran_command='gfortran'):
    """Build bader package.

    Args:
        fortran_command: The Fortran compiler command.
    """
    bader_url = 'http://theory.cm.utexas.edu/henkelman/code/bader/download/bader.tar.gz'
    cwd = os.getcwd()
    state = True
    try:
        urlretrieve(bader_url, 'bader.tar.gz')
        subprocess.call(['tar', '-zxf', 'bader.tar.gz'])
        os.chdir('bader')
        subprocess.call(['cp', 'makefile.osx_' + fortran_command, 'makefile'])
        subprocess.call(['make'])
        shutil.copy('bader', os.path.join('..', 'bader_exe'))
        os.chdir('..')
        shutil.rmtree('bader')
        os.remove('bader.tar.gz')
        shutil.move('bader_exe', 'bader')
    except Exception as exc:
        print(str(exc))
        state = False
    finally:
        os.chdir(cwd)
    return state