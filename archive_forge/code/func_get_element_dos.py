from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
def get_element_dos(self) -> dict[SpeciesLike, Dos]:
    """Get element projected Dos.

        Returns:
            dict[Element, Dos]
        """
    el_dos = {}
    for site, atom_dos in self.pdos.items():
        el = site.specie
        for pdos in atom_dos.values():
            if el not in el_dos:
                el_dos[el] = pdos
            else:
                el_dos[el] = add_densities(el_dos[el], pdos)
    return {el: Dos(self.efermi, self.energies, densities) for el, densities in el_dos.items()}