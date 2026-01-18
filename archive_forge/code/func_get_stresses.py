from abc import ABC, abstractmethod
from typing import Mapping, Any
def get_stresses(self, atoms=None):
    """the calculator should return intensive stresses, i.e., such that
                stresses.sum(axis=0) == stress
        """
    return self.get_property('stresses', atoms)