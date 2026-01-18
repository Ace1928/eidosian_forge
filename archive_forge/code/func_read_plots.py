from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_plots(string: str) -> dict:
    """
        Read plots parameters from string.

        Args:
            string (str): String

        Returns:
            dict[str, str]: plots parameters.
        """
    header = '^\\s*\\$plots'
    row = '\\s*([a-zA-Z\\_]+)\\s+(\\S+)'
    footer = '^\\s*\\$end'
    plots_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if plots_table == []:
        print("No valid plots inputs found. Note that there should be no '=' characters in plots input lines.")
        return {}
    return dict(plots_table[0])