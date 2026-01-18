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
def configure_pmg(args: Namespace):
    """Handle configure command."""
    if args.potcar_dirs:
        setup_potcars(args.potcar_dirs)
    elif args.install:
        install_software(args.install)
    elif args.var_spec:
        add_config_var(args.var_spec, args.backup)
    elif args.cp2k_data_dirs:
        setup_cp2k_data(args.cp2k_data_dirs)