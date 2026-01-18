import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def converge_to_eigenmode(self):
    """Perform an eigenmode search."""
    self.set_up_for_eigenmode_search()
    stoprot = False
    f_rot_min = self.control.get_parameter('f_rot_min')
    f_rot_max = self.control.get_parameter('f_rot_max')
    trial_angle = self.control.get_parameter('trial_angle')
    max_num_rot = self.control.get_parameter('max_num_rot')
    extrapolate = self.control.get_parameter('extrapolate_forces')
    while not stoprot:
        if self.forces1E is None:
            self.update_virtual_forces()
        else:
            self.update_virtual_forces(extrapolated_forces=True)
        self.forces1A = self.forces1
        self.update_curvature()
        f_rot_A = self.get_rotational_force()
        if norm(f_rot_A) <= f_rot_min:
            self.log(f_rot_A, None)
            stoprot = True
        else:
            n_A = self.eigenmode
            rot_unit_A = normalize(f_rot_A)
            c0 = self.get_curvature()
            c0d = np.vdot(self.forces2 - self.forces1, rot_unit_A) / self.dR
            n_B, rot_unit_B = rotate_vectors(n_A, rot_unit_A, trial_angle)
            self.eigenmode = n_B
            self.update_virtual_forces()
            self.forces1B = self.forces1
            c1d = np.vdot(self.forces2 - self.forces1, rot_unit_B) / self.dR
            a1 = c0d * cos(2 * trial_angle) - c1d / (2 * sin(2 * trial_angle))
            b1 = 0.5 * c0d
            a0 = 2 * (c0 - a1)
            rotangle = atan(b1 / a1) / 2.0
            cmin = a0 / 2.0 + a1 * cos(2 * rotangle) + b1 * sin(2 * rotangle)
            if c0 < cmin:
                rotangle += pi / 2.0
            n_min, dummy = rotate_vectors(n_A, rot_unit_A, rotangle)
            self.update_eigenmode(n_min)
            self.update_curvature(cmin)
            self.log(f_rot_A, rotangle)
            if extrapolate:
                self.forces1E = sin(trial_angle - rotangle) / sin(trial_angle) * self.forces1A + sin(rotangle) / sin(trial_angle) * self.forces1B + (1 - cos(rotangle) - sin(rotangle) * tan(trial_angle / 2.0)) * self.forces0
            else:
                self.forces1E = None
        if not stoprot:
            if self.control.get_counter('rotcount') >= max_num_rot:
                stoprot = True
            elif norm(f_rot_A) <= f_rot_max:
                stoprot = True