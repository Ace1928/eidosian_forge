from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_pcm(string: str) -> dict:
    """
        Read pcm parameters from string.

        Args:
            string (str): String

        Returns:
            dict[str, str]: PCM parameters
        """
    header = '^\\s*\\$pcm'
    row = '\\s*([a-zA-Z\\_]+)\\s+(\\S+)'
    footer = '^\\s*\\$end'
    pcm_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    if not pcm_table:
        print("No valid PCM inputs found. Note that there should be no '=' characters in PCM input lines.")
        return {}
    return dict(pcm_table[0])