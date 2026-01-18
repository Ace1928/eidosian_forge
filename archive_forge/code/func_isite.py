from __future__ import annotations
import abc
from monty.json import MSONable
@property
def isite(self):
    """Index of the central site."""
    return self.i_central_site