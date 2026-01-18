from __future__ import annotations
import abc
from collections import namedtuple
from collections.abc import Iterable
from enum import Enum, unique
from pprint import pformat
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.collections import AttrDict
from monty.design_patterns import singleton
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.core import ArrayWithUnit, Lattice, Species, Structure, units
def lattice_from_abivars(cls=None, *args, **kwargs):
    """
    Returns a `Lattice` object from a dictionary
    with the Abinit variables `acell` and either `rprim` in Bohr or `angdeg`
    If acell is not given, the Abinit default is used i.e. [1,1,1] Bohr.

    Args:
        cls: Lattice class to be instantiated. Defaults to pymatgen.core.Lattice.

    Example:
        lattice_from_abivars(acell=3*[10], rprim=np.eye(3))
    """
    cls = cls or Lattice
    kwargs.update(dict(*args))
    r_prim = kwargs.get('rprim')
    ang_deg = kwargs.get('angdeg')
    a_cell = kwargs['acell']
    if r_prim is not None:
        if ang_deg is not None:
            raise ValueError('angdeg and rprimd are mutually exclusive')
        r_prim = np.reshape(r_prim, (3, 3))
        rprimd = [float(a_cell[i]) * r_prim[i] for i in range(3)]
        return cls(ArrayWithUnit(rprimd, 'bohr').to('ang'))
    if ang_deg is not None:
        ang_deg = np.reshape(ang_deg, 3)
        if np.any(ang_deg <= 0.0):
            raise ValueError(f'Angles must be > 0 but got {ang_deg}')
        if ang_deg.sum() >= 360.0:
            raise ValueError(f'The sum of angdeg must be lower than 360, ang_deg={ang_deg!r}')
        tol12 = 1e-12
        pi, sin, cos, sqrt = (np.pi, np.sin, np.cos, np.sqrt)
        r_prim = np.zeros((3, 3))
        if abs(ang_deg[0] - ang_deg[1]) < tol12 and abs(ang_deg[1] - ang_deg[2]) < tol12 and (abs(ang_deg[0] - 90.0) + abs(ang_deg[1] - 90.0) + abs(ang_deg[2] - 90) > tol12):
            cos_ang = cos(pi * ang_deg[0] / 180.0)
            a2 = 2.0 / 3.0 * (1.0 - cos_ang)
            aa = sqrt(a2)
            cc = sqrt(1.0 - a2)
            r_prim[0, 0] = aa
            r_prim[0, 1] = 0.0
            r_prim[0, 2] = cc
            r_prim[1, 0] = -0.5 * aa
            r_prim[1, 1] = sqrt(3.0) * 0.5 * aa
            r_prim[1, 2] = cc
            r_prim[2, 0] = -0.5 * aa
            r_prim[2, 1] = -sqrt(3.0) * 0.5 * aa
            r_prim[2, 2] = cc
        else:
            r_prim[0, 0] = 1.0
            r_prim[1, 0] = cos(pi * ang_deg[2] / 180.0)
            r_prim[1, 1] = sin(pi * ang_deg[2] / 180.0)
            r_prim[2, 0] = cos(pi * ang_deg[1] / 180.0)
            r_prim[2, 1] = (cos(pi * ang_deg[0] / 180.0) - r_prim[1, 0] * r_prim[2, 0]) / r_prim[1, 1]
            r_prim[2, 2] = sqrt(1.0 - r_prim[2, 0] ** 2 - r_prim[2, 1] ** 2)
        rprimd = [float(a_cell[i]) * r_prim[i] for i in range(3)]
        return cls(ArrayWithUnit(rprimd, 'bohr').to('ang'))
    raise ValueError(f"Don't know how to construct a Lattice from dict:\n{pformat(kwargs)}")