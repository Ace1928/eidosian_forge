from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def cdft_template(cdft: list[list[dict]]) -> str:
    """
        Args:
            cdft: list of lists of dicts.

        Returns:
            str: CDFT section.
        """
    cdft_list = []
    cdft_list.append('$cdft')
    for ii, state in enumerate(cdft, start=1):
        for constraint in state:
            types = constraint['types']
            cdft_list.append(f'   {constraint['value']}')
            type_strings = []
            for typ in types:
                if typ is None or typ.lower() in ['c', 'charge']:
                    type_strings.append('')
                elif typ.lower() in ['s', 'spin']:
                    type_strings.append('s')
                else:
                    raise ValueError('Invalid CDFT constraint type!')
            for coef, first, last, type_string in zip(constraint['coefficients'], constraint['first_atoms'], constraint['last_atoms'], type_strings):
                if type_string != '':
                    cdft_list.append(f'   {coef} {first} {last} {type_string}')
                else:
                    cdft_list.append(f'   {coef} {first} {last}')
        if len(cdft) != 1 and ii < len(state):
            cdft_list.append('--------------')
    if cdft_list[-1] == '--------------':
        del cdft_list[-1]
    cdft_list.append('$end')
    return '\n'.join(cdft_list)