from __future__ import annotations
from itertools import chain, combinations
from pymatgen.core import Element
from pymatgen.core.composition import Composition
def obtain_band_edges(self):
    """Fill up the atomic orbitals with available electrons.

        Returns:
            HOMO, LUMO, and whether it's a metal.
        """
    orbitals = self.aos_as_list()
    electrons = Composition(self.composition).total_electrons
    partial_filled = []
    for orbital in orbitals:
        if electrons <= 0:
            break
        if 's' in orbital[1]:
            electrons += -2
        elif 'p' in orbital[1]:
            electrons += -6
        elif 'd' in orbital[1]:
            electrons += -10
        elif 'f' in orbital[1]:
            electrons += -14
        partial_filled.append(orbital)
    if electrons != 0:
        homo = partial_filled[-1]
        lumo = partial_filled[-1]
    else:
        homo = partial_filled[-1]
        try:
            lumo = orbitals[len(partial_filled)]
        except Exception:
            lumo = None
    return {'HOMO': homo, 'LUMO': lumo, 'metal': homo == lumo}