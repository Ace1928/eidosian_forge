from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_svp(string: str) -> dict:
    """Read svp parameters from string."""
    header = '^\\s*\\$svp'
    row = '(\\w.*)\\n'
    footer = '^\\s*\\$end'
    svp_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if svp_table == []:
        print('No valid svp inputs found.')
        return {}
    svp_list = svp_table[0][0][0].split(', ')
    svp_dict = {}
    for s in svp_list:
        svp_dict[s.split('=')[0]] = s.split('=')[1]
    return svp_dict