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
def setup_potcars(potcar_dirs: list[str]):
    """Setup POTCAR directories."""
    psp_dir, target_dir = (os.path.abspath(d) for d in potcar_dirs)
    try:
        os.makedirs(target_dir)
    except OSError:
        reply = input('Destination directory exists. Continue (y/n)? ')
        if reply != 'y':
            raise SystemExit('Exiting ...')
    print('Generating pymatgen resources directory...')
    name_mappings = {'potpaw_PBE': 'POT_GGA_PAW_PBE', 'potpaw_PBE_52': 'POT_GGA_PAW_PBE_52', 'potpaw_PBE_54': 'POT_GGA_PAW_PBE_54', 'potpaw_PBE.52': 'POT_GGA_PAW_PBE_52', 'potpaw_PBE.54': 'POT_GGA_PAW_PBE_54', 'potpaw_LDA': 'POT_LDA_PAW', 'potpaw_LDA.52': 'POT_LDA_PAW_52', 'potpaw_LDA.54': 'POT_LDA_PAW_54', 'potpaw_LDA_52': 'POT_LDA_PAW_52', 'potpaw_LDA_54': 'POT_LDA_PAW_54', 'potUSPP_LDA': 'POT_LDA_US', 'potpaw_GGA': 'POT_GGA_PAW_PW91', 'potUSPP_GGA': 'POT_GGA_US_PW91'}
    for parent, subdirs, _files in os.walk(psp_dir):
        basename = os.path.basename(parent)
        basename = name_mappings.get(basename, basename)
        for subdir in subdirs:
            filenames = glob(os.path.join(parent, subdir, 'POTCAR*'))
            if len(filenames) > 0:
                try:
                    base_dir = os.path.join(target_dir, basename)
                    os.makedirs(base_dir, exist_ok=True)
                    fname = filenames[0]
                    dest = os.path.join(base_dir, os.path.basename(fname))
                    shutil.copy(fname, dest)
                    ext = fname.split('.')[-1]
                    if ext.upper() in ['Z', 'GZ']:
                        with subprocess.Popen(['gunzip', dest]) as p:
                            p.communicate()
                    elif ext.upper() == 'BZ2':
                        with subprocess.Popen(['bunzip2', dest]) as p:
                            p.communicate()
                    if subdir == 'Osmium':
                        subdir = 'Os'
                    dest = os.path.join(base_dir, f'POTCAR.{subdir}')
                    shutil.move(f'{base_dir}/POTCAR', dest)
                    with subprocess.Popen(['gzip', '-f', dest]) as p:
                        p.communicate()
                except Exception as exc:
                    print(f'An error has occurred. Message is {exc}. Trying to continue... ')
    print(f"\nPSP resources directory generated. It is recommended that you run 'pmg config --add PMG_VASP_PSP_DIR {os.path.abspath(target_dir)}'")
    print('Start a new terminal to ensure that your environment variables are properly set.')