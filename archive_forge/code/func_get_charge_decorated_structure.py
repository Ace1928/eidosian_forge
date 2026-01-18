from __future__ import annotations
import os
import shutil
import subprocess
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.dev import deprecated
from monty.shutil import decompress_file
from monty.tempfile import ScratchDir
from pymatgen.io.common import VolumetricData
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def get_charge_decorated_structure(self) -> Structure:
    """Returns a charge decorated structure.

        Note, this assumes that the Bader analysis was correctly performed on a file
        with electron densities
        """
    charges = [-self.get_charge(i) for i in range(len(self.structure))]
    struct = self.structure.copy()
    struct.add_site_property('charge', charges)
    return struct