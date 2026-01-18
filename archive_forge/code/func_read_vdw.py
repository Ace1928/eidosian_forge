from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_vdw(string: str) -> tuple[str, dict]:
    """
        Read van der Waals parameters from string.

        Args:
            string (str): String

        Returns:
            tuple[str, dict]: (vdW mode ('atomic' or 'sequential'), dict of van der Waals radii)
        """
    header = '^\\s*\\$van_der_waals'
    row = '[^\\d]*(\\d+).?(\\d+.\\d+)?.*'
    footer = '^\\s*\\$end'
    vdw_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if not vdw_table:
        print("No valid vdW inputs found. Note that there should be no '=' characters in vdW input lines.")
        return ('', {})
    mode = 'sequential' if vdw_table[0][0][0] == 2 else 'atomic'
    return (mode, dict(vdw_table[0][1:]))