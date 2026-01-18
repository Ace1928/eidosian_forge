from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@classmethod
def from_multi_jobs_file(cls, filename: str) -> list[Self]:
    """
        Create list of QcInput from a file.

        Args:
            filename (str): Filename

        Returns:
            List of QCInput objects
        """
    with zopen(filename, mode='rt') as file:
        multi_job_strings = file.read().split('@@@')
        return [cls.from_str(i) for i in multi_job_strings]