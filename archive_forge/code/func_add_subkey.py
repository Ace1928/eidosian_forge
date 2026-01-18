from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def add_subkey(self, subkey):
    """
        Add a new subkey to this key.

        Args:
            subkey (AdfKey): A new subkey.

        Notes:
            Duplicate check will not be performed if this is an 'Atoms' block.
        """
    if self.key.lower() == 'atoms' or not self.has_subkey(subkey):
        self.subkeys.append(subkey)