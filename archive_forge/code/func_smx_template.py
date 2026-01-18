from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def smx_template(smx: dict) -> str:
    """
        Args:
            smx ():

        Returns:
            str: Solvation model with short-range corrections.
        """
    smx_list = []
    smx_list.append('$smx')
    for key, value in smx.items():
        if value == 'tetrahydrofuran':
            smx_list.append(f'   {key} thf')
        elif value == 'dimethyl sulfoxide':
            smx_list.append(f'   {key} dmso')
        else:
            smx_list.append(f'   {key} {value}')
    smx_list.append('$end')
    return '\n'.join(smx_list)