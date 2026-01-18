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
def temp_decompress(file: str | Path, target_dir: str='.') -> str:
    """Utility function to copy a compressed file to a target directory (ScratchDir)
            and decompress it, to avoid modifying files in place.

            Args:
                file (str | Path): The path to the compressed file to be decompressed.
                target_dir (str, optional): The target directory where the decompressed file will be stored.
                    Defaults to "." (current directory).

            Returns:
                str: The path to the decompressed file if successful.
            """
    file = Path(file)
    if file.suffix.lower() in {'.bz2', '.gz', '.z'}:
        shutil.copy(file, f'{target_dir}/{file.name}')
        out_file = decompress_file(f'{target_dir}/{file.name}')
        if file:
            return str(out_file)
        raise FileNotFoundError(f'File {out_file} not found.')
    return str(file)