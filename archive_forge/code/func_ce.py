from __future__ import annotations
import abc
from monty.json import MSONable
@property
def ce(self):
    """Coordination environment of this node."""
    return self.coordination_environment