from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
def contains_command(self, command: str, stage_name: str | None=None) -> bool:
    """
        Returns whether a given command is present in the LammpsInputFile.
        A stage name can be given; in this case the search will happen only for this stage.

        Args:
            command (str): String with the command to find in the input file (e.g., "units").
            stage_name (str): String giving the stage name where the change should take place.

        Returns:
            bool: True if the command is present, False if not.
        """
    return bool(self.get_args(command, stage_name))