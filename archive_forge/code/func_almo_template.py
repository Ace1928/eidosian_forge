from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def almo_template(almo_coupling: list[list[tuple[int, int]]]) -> str:
    """
        Args:
            almo: list of lists of int 2-tuples.

        Returns:
            str: ALMO coupling section.
        """
    almo_list = []
    almo_list.append('$almo_coupling')
    if len(almo_coupling) != 2:
        raise ValueError('ALMO coupling calculations require exactly two states!')
    state_1 = almo_coupling[0]
    state_2 = almo_coupling[1]
    for frag in state_1:
        almo_list.append(f'   {int(frag[0])} {int(frag[1])}')
    almo_list.append('   --')
    for frag in state_2:
        almo_list.append(f'   {int(frag[0])} {int(frag[1])}')
    almo_list.append('$end')
    return '\n'.join(almo_list)