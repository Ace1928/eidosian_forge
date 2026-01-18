from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def pcm_nonels_template(pcm_nonels: dict) -> str:
    """
        Template for the $pcm_nonels section.

        Arg
            pcm_nonels: dict of CMIRS parameters, e.g.
            {
                "a": "-0.006736",
                "b": "0.032698",
                "c": "-1249.6",
                "d": "-21.405",
                "gamma": "3.7",
                "solvrho": "0.05",
                "delta": 7,
                "gaulag_n": 40,
            }

        Returns:
            str: the $pcm_nonels section. Note that all parameters will be concatenated onto
                a single line formatted as a FORTRAN namelist. This is necessary
                because the non-electrostatic part of the CMIRS solvation model in Q-Chem
                calls a secondary code.
        """
    pcm_nonels_list = []
    pcm_nonels_list.append('$pcm_nonels')
    for key, value in pcm_nonels.items():
        if value is not None:
            pcm_nonels_list.append(f'   {key} {value}')
    pcm_nonels_list.append('$end')
    return '\n'.join(pcm_nonels_list)