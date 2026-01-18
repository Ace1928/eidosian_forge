from abc import ABC, abstractmethod
from typing import Mapping, Any
def get_magnetic_moments(self, atoms=None):
    """Calculate magnetic moments projected onto atoms."""
    return self.get_property('magmoms', atoms)