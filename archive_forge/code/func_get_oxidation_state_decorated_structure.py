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
def get_oxidation_state_decorated_structure(self, nelects: list[int] | None=None) -> Structure:
    """Returns an oxidation state decorated structure based on bader analysis results.
        Each site is assigned a charge based on the computed partial atomic charge from bader.

        Note, this assumes that the Bader analysis was correctly performed on a file
        with electron densities.

        Args:
            nelects (list[int]): number of electrons associated with an isolated atom at this index.

        Returns:
            Structure: with bader-analysis-based oxidation states.
        """
    struct = self.structure.copy()
    charges = [self.get_partial_charge(idx, None if not nelects else nelects[idx]) for idx in range(len(struct))]
    struct.add_oxidation_state_by_site(charges)
    return struct