from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def solvent_template(solvent: dict) -> str:
    """
        Solvent template.

        Args:
            solvent ():

        Returns:
            str: Solvent section.
        """
    solvent_list = []
    solvent_list.append('$solvent')
    for key, value in solvent.items():
        solvent_list.append(f'   {key} {value}')
    solvent_list.append('$end')
    return '\n'.join(solvent_list)