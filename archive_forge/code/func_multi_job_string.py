from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def multi_job_string(job_list: list[QCInput]) -> str:
    """
        Args:
            job_list (): List of jobs.

        Returns:
            str: String representation of a multi-job input file.
        """
    multi_job_string = ''
    for i, job_i in enumerate(job_list, start=1):
        if i < len(job_list):
            multi_job_string += str(job_i) + '\n@@@\n\n'
        else:
            multi_job_string += str(job_i)
    return multi_job_string