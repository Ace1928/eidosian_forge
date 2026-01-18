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
def remove_stage(self, stage_name: str) -> None:
    """
        Removes a whole stage from the LammpsInputFile.

        Args:
            stage_name (str): name of the stage to remove.
        """
    if stage_name in self.stages_names:
        idx = self.stages_names.index(stage_name)
        self.stages.pop(idx)
    else:
        raise LookupError('The given stage name is not present in this LammpsInputFile.')