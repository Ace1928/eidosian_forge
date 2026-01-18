from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def scan_template(scan: dict[str, list]) -> str:
    """
        Args:
            scan (dict): Dictionary with scan section information.
                Ex: {"stre": ["3 6 1.5 1.9 0.1"], "tors": ["1 2 3 4 -180 180 15"]}.

        Returns:
            String representing Q-Chem input format for scan section
        """
    scan_list = []
    scan_list.append('$scan')
    total_vars = sum((len(v) for v in scan.values()))
    if total_vars > 2:
        raise ValueError('Q-Chem only supports PES_SCAN with two or less variables.')
    for var_type, variables in scan.items():
        if variables not in [None, []]:
            for var in variables:
                scan_list.append(f'   {var_type} {var}')
    scan_list.append('$end')
    return '\n'.join(scan_list)