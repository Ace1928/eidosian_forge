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
def displace(self, displacement_vector=None, mask=None, method=None, displacement_center=None, radius=None, number_of_atoms=None, gauss_std=None, mic=True, log=True):
    """Move the atoms away from their current position.

        This is one of the essential parts of minimum mode searches.
        The parameters can all be set in the control object and overwritten
        when this method is run, apart from *displacement_vector*.
        It is preferred to modify the control values rather than those here
        in order for the correct ones to show up in the log file.

        *method* can be either 'gauss' for random displacement or 'vector'
        to perform a predefined displacement.

        *gauss_std* is the standard deviation of the gauss curve that is
        used for random displacement.

        *displacement_center* can be either the number of an atom or a 3D
        position. It must be accompanied by a *radius* (all atoms within it
        will be displaced) or a *number_of_atoms* which decides how many of
        the closest atoms will be displaced.

        *mic* controls the usage of the Minimum Image Convention.

        If both *mask* and *displacement_center* are used, the atoms marked
        as False in the *mask* will not be affected even though they are
        within reach of the *displacement_center*.

        The parameters priority order:
        1) displacement_vector
        2) mask
        3) displacement_center (with radius and/or number_of_atoms)

        If both *radius* and *number_of_atoms* are supplied with
        *displacement_center*, only atoms that fulfill both criteria will
        be displaced.

        """
    if mask is None:
        mask = self.control.get_parameter('mask')
    if method is None:
        method = self.control.get_parameter('displacement_method')
    if gauss_std is None:
        gauss_std = self.control.get_parameter('gauss_std')
    if displacement_center is None:
        displacement_center = self.control.get_parameter('displacement_center')
    if radius is None:
        radius = self.control.get_parameter('displacement_radius')
    if number_of_atoms is None:
        number_of_atoms = self.control.get_parameter('number_of_displacement_atoms')
    if displacement_vector is not None and method.lower() != 'vector':
        e = 'displacement_vector was supplied but a different method ' + "('%s') was chosen.\n" % str(method)
        raise ValueError(e)
    elif displacement_vector is None and method.lower() == 'vector':
        e = 'A displacement_vector must be supplied when using ' + "method = '%s'.\n" % str(method)
        raise ValueError(e)
    elif displacement_center is not None and radius is None and (number_of_atoms is None):
        e = 'When displacement_center is chosen, either radius or ' + 'number_of_atoms must be supplied.\n'
        raise ValueError(e)
    if displacement_center is not None:
        c = displacement_center
        if isinstance(c, int):
            c = displacement_center % len(self)
            d = [(k, self.get_distance(k, c, mic=mic)) for k in range(len(self))]
        elif len(c) == 3 and [type(c_k) for c_k in c] == [float] * 3:
            d = [(k, norm(self.get_positions()[k] - c)) for k in range(len(self))]
        else:
            e = 'displacement_center must be either the number of an ' + 'atom in MinModeAtoms object or a 3D position ' + '(3-tuple of floats).'
            raise ValueError(e)
        if radius is not None:
            r_mask = [dist[1] < radius for dist in d]
        else:
            r_mask = [True for _ in range(len(self))]
        if number_of_atoms is not None:
            d_sorted = [n[0] for n in sorted(d, key=lambda k: k[1])]
            n_nearest = d_sorted[:number_of_atoms]
            n_mask = [k in n_nearest for k in range(len(self))]
        else:
            n_mask = [True for _ in range(len(self))]
        c_mask = [n_mask[k] and r_mask[k] for k in range(len(self))]
    else:
        c_mask = None
    if mask is None:
        mask = [True for _ in range(len(self))]
        if c_mask is None:
            w = 'It was not possible to figure out which atoms to ' + 'displace, Will try to displace all atoms.\n'
            warnings.warn(w, UserWarning)
            if self.logfile is not None:
                self.logfile.write('MINMODE:WARN: ' + w + '\n')
                self.logfile.flush()
    if c_mask is not None:
        mask = [mask[k] and c_mask[k] for k in range(len(self))]
    if displacement_vector is None:
        displacement_vector = []
        for k in range(len(self)):
            if mask[k]:
                diff_line = []
                for _ in range(3):
                    if method.lower() == 'gauss':
                        if not gauss_std:
                            gauss_std = self.control.get_parameter('gauss_std')
                        diff = self.random_state.normal(0.0, gauss_std)
                    else:
                        e = 'Invalid displacement method >>%s<<' % str(method)
                        raise ValueError(e)
                    diff_line.append(diff)
                displacement_vector.append(diff_line)
            else:
                displacement_vector.append([0.0] * 3)
    for k in range(len(mask)):
        if not mask[k]:
            displacement_vector[k] = [0.0] * 3
    if log:
        pos0 = self.get_positions()
    self.set_positions(self.get_positions() + displacement_vector)
    if log:
        parameters = {'mask': mask, 'displacement_method': method, 'gauss_std': gauss_std, 'displacement_center': displacement_center, 'displacement_radius': radius, 'number_of_displacement_atoms': number_of_atoms}
        self.displacement_log(self.get_positions() - pos0, parameters)