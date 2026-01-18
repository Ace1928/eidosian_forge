from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def find_sections(string: str) -> list:
    """
        Find sections in the string.

        Args:
            string (str): String

        Returns:
            List of sections.
        """
    patterns = {'sections': '^\\s*?\\$([a-z_]+)', 'multiple_jobs': '(@@@)'}
    matches = read_pattern(string, patterns)
    sections = [val[0] for val in matches['sections']]
    sections = [sec for sec in sections if sec != 'end']
    if 'multiple_jobs' in matches:
        raise ValueError('Output file contains multiple qchem jobs please parse separately')
    if 'molecule' not in sections:
        raise ValueError('Output file does not contain a molecule section')
    if 'rem' not in sections:
        raise ValueError('Output file does not contain a REM section')
    return sections