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
@classmethod
def gamma_centered(cls, kpts=(1, 1, 1), use_symmetries=True, use_time_reversal=True):
    """
        Convenient static constructor for an automatic Gamma centered Kpoint grid.

        Args:
            kpts: Subdivisions N_1, N_2 and N_3 along reciprocal lattice vectors.
            use_symmetries: False if spatial symmetries should not be used
                to reduce the number of independent k-points.
            use_time_reversal: False if time-reversal symmetry should not be used
                to reduce the number of independent k-points.

        Returns:
            KSampling object.
        """
    return cls(kpts=[kpts], kpt_shifts=(0.0, 0.0, 0.0), use_symmetries=use_symmetries, use_time_reversal=use_time_reversal, comment='gamma-centered mode')