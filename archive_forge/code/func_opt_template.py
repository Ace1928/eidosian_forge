from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def opt_template(opt: dict[str, list]) -> str:
    """
        Optimization template.

        Args:
            opt ():

        Returns:
            str: Optimization template.
        """
    opt_list = []
    opt_list.append('$opt')
    for key, value in opt.items():
        opt_list.append(f'{key}')
        for i in value:
            opt_list.append(f'   {i}')
        opt_list.extend((f'END{key}', ''))
    del opt_list[-1]
    opt_list.append('$end')
    return '\n'.join(opt_list)