from __future__ import annotations
import abc
from monty.json import MSONable
@property
def atom_symbol(self):
    """Symbol of the atom on the central site."""
    return self.central_site.specie.symbol