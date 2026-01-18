from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_almo(string: str) -> list[list[tuple[int, int]]]:
    """
        Read ALMO coupling parameters from string.

        Args:
            string (str): String

        Returns:
            list[list[tuple[int, int]]]: ALMO coupling parameters
        """
    pattern = {'key': '\\$almo_coupling\\s*\\n((?:\\s*[\\-0-9]+\\s+[\\-0-9]+\\s*\\n)+)\\s*\\-\\-((?:\\s*[\\-0-9]+\\s+[\\-0-9]+\\s*\\n)+)\\s*\\$end'}
    section = read_pattern(string, pattern)['key']
    if len(section) == 0:
        print('No valid almo inputs found.')
        return []
    section = section[0]
    almo_coupling = [[], []]
    state_1 = section[0]
    for line in state_1.strip().split('\n'):
        contents = line.split()
        almo_coupling[0].append((int(contents[0]), int(contents[1])))
    state_2 = section[1]
    for line in state_2.strip().split('\n'):
        contents = line.split()
        almo_coupling[1].append((int(contents[0]), int(contents[1])))
    return almo_coupling