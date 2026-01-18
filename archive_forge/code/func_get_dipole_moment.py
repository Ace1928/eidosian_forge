from abc import ABC, abstractmethod
from typing import Mapping, Any
def get_dipole_moment(self, atoms=None):
    return self.get_property('dipole', atoms)