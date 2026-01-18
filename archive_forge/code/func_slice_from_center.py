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
def slice_from_center(data: np.ndarray, x_width: int, y_width: int, z_width: int) -> np.ndarray:
    """Slices a central window from the data array."""
    x, y, z = data.shape
    start_x = x // 2 - x_width // 2
    start_y = y // 2 - y_width // 2
    start_z = z // 2 - z_width // 2
    return data[start_x:start_x + x_width, start_y:start_y + y_width, start_z:start_z + z_width]