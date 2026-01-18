from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
def remove_subkey(self, subkey):
    """
        Remove the given subkey, if existed, from this AdfKey.

        Args:
            subkey (str or AdfKey): The subkey to remove.
        """
    if len(self.subkeys) > 0:
        key = subkey if isinstance(subkey, str) else subkey.key
        for idx, subkey in enumerate(self.subkeys):
            if subkey.key == key:
                self.subkeys.pop(idx)
                break