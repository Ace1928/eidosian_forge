from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def is_block_key(self) -> bool:
    """Return True if this key is a block key."""
    return self.name.upper() in self.block_keys