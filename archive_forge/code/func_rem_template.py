from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def rem_template(rem: dict) -> str:
    """
        Args:
            rem ():

        Returns:
            str: REM template.
        """
    rem_list = []
    rem_list.append('$rem')
    for key, value in rem.items():
        rem_list.append(f'   {key} = {value}')
    rem_list.append('$end')
    return '\n'.join(rem_list)