from __future__ import annotations
from enum import Enum, unique
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
def get_00t_magmom_with_xyz_saxis(self):
    """For internal implementation reasons, in non-collinear calculations VASP prefers the following.

            MAGMOM = 0 0 total_magnetic_moment
            SAXIS = x y z

        to an equivalent:

            MAGMOM = x y z
            SAXIS = 0 0 1

        This method returns a Magmom object with magnetic moment [0, 0, t],
        where t is the total magnetic moment, and saxis rotated as required.

        A consistent direction of saxis is applied such that t might be positive
        or negative depending on the direction of the initial moment. This is useful
        in the case of collinear structures, rather than constraining assuming
        t is always positive.

        Returns:
            Magmom
        """
    ref_direction = np.array([1.01, 1.02, 1.03])
    t = abs(self)
    if t != 0:
        new_saxis = self.moment / np.linalg.norm(self.moment)
        if np.dot(ref_direction, new_saxis) < 0:
            t = -t
            new_saxis = -new_saxis
        return Magmom([0, 0, t], saxis=new_saxis)
    return Magmom(self)