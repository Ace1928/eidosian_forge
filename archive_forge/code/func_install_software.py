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
def install_software(install: Literal['enumlib', 'bader']):
    """Install all optional external software."""
    try:
        subprocess.call(['ifort', '--version'])
        print('Found ifort')
        fortran_command = 'ifort'
    except Exception:
        try:
            subprocess.call(['gfortran', '--version'])
            print('Found gfortran')
            fortran_command = 'gfortran'
        except Exception as exc:
            print(str(exc))
            raise SystemExit('No fortran compiler found.')
    enum = bader = None
    if install == 'enumlib':
        print('Building enumlib')
        enum = build_enum(fortran_command)
        print()
    elif install == 'bader':
        print('Building bader')
        bader = build_bader(fortran_command)
        print()
    if bader or enum:
        print(f'Please add {os.path.abspath('.')} to your PATH or move the executables multinum.x, makestr.x and/or bader to a location in your PATH.\n')