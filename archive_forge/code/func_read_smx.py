from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_smx(string: str) -> dict:
    """
        Read smx parameters from string.

        Args:
            string (str): String

        Returns:
            dict[str, str] SMX parameters.
        """
    header = '^\\s*\\$smx'
    row = '\\s*([a-zA-Z\\_]+)\\s+(\\S+)'
    footer = '^\\s*\\$end'
    smx_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if not smx_table:
        print("No valid smx inputs found. Note that there should be no '=' characters in smx input lines.")
        return {}
    smx = dict(smx_table[0])
    if smx['solvent'] == 'tetrahydrofuran':
        smx['solvent'] = 'thf'
    elif smx['solvent'] == 'dimethyl sulfoxide':
        smx['solvent'] = 'dmso'
    return smx