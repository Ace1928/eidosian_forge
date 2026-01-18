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
def find_encompassing_vol(data: np.ndarray) -> np.ndarray | None:
    """Find the central encompassing volume which
            holds all the data within a precision.
            """
    total = np.sum(data)
    for idx in range(np.max(data.shape)):
        sliced_data = slice_from_center(data, idx, idx, idx)
        if total - np.sum(sliced_data) < 0.1:
            return sliced_data
    return None