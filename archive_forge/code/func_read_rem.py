from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_rem(string: str) -> dict:
    """
        Parse rem from string.

        Args:
            string (str): String

        Returns:
            dict[str, str]: REM section
        """
    header = '^\\s*\\$rem'
    row = '\\s*([a-zA-Z\\_\\d]+)\\s*=?\\s*(\\S+)'
    footer = '^\\s*\\$end'
    rem_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    return dict(rem_table[0])