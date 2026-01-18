from typing import Tuple
import numpy as np
from cvxpy.atoms.affine.affine_atom import AffAtom
def is_imag(self) -> bool:
    """Is the expression imaginary?
        """
    return False