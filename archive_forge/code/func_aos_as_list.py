from __future__ import annotations
from itertools import chain, combinations
from pymatgen.core import Element
from pymatgen.core.composition import Composition
def aos_as_list(self):
    """The orbitals energies in eV are represented as
        [['O', '1s', -18.758245], ['O', '2s', -0.871362], ['O', '2p', -0.338381]]
        Data is obtained from
        https://www.nist.gov/pml/data/atomic-reference-data-electronic-structure-calculations.

        Returns:
            A list of atomic orbitals, sorted from lowest to highest energy.
        """
    return sorted(chain.from_iterable([self.aos[el] * int(self.composition[el]) for el in self.elements]), key=lambda x: x[2])