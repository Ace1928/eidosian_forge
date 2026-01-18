from __future__ import annotations
import os
from monty.design_patterns import singleton
from pymatgen.core import Composition, Element
@staticmethod
def get_hhi_designation(hhi):
    """
        Gets a designation for low, medium, high HHI, as specified in "U.S.
        Department of Justice and the Federal Trade Commission, Horizontal
        merger guidelines; 2010.".

        Args:
            hhi (float): HHI value

        Returns:
            The designation as String
        """
    if hhi is None:
        return None
    if 0 <= hhi < 1500:
        return 'low'
    if 1500 <= hhi <= 2500:
        return 'medium'
    return 'high'