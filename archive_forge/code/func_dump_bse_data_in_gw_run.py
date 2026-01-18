from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
def dump_bse_data_in_gw_run(self, BSE_dump=True):
    """
        Args:
            BSE_dump: boolean

        Returns:
            set the "do_bse" variable to one in cell.in
        """
    if BSE_dump:
        self.bse_tddft_options.update(do_bse=1, do_tddft=0)
    else:
        self.bse_tddft_options.update(do_bse=0, do_tddft=0)