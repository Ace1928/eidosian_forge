from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_geom_opt(string: str) -> dict:
    """
        Read geom_opt parameters from string.

        Args:
            string (str): String

        Returns:
            dict[str, str]: geom_opt parameters.
        """
    header = '^\\s*\\$geom_opt'
    row = '\\s*([a-zA-Z\\_]+)\\s*=?\\s*(\\S+)'
    footer = '^\\s*\\$end'
    geom_opt_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if geom_opt_table == []:
        print('No valid geom_opt inputs found.')
        return {}
    return dict(geom_opt_table[0])