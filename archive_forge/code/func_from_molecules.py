from __future__ import annotations
import itertools
import warnings
from collections.abc import Iterator, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Composition, DummySpecies, Element, Lattice, Molecule, Species, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
@classmethod
def from_molecules(cls, molecules: list[Molecule], **kwargs) -> Self:
    """Create trajectory from a list of molecules.

        Note: Assumes no atoms removed during simulation.

        Args:
            molecules: pymatgen Molecule objects.
            **kwargs: Additional kwargs passed to Trajectory constructor.

        Returns:
            A trajectory from the structures.
        """
    species = molecules[0].species
    coords = [mol.cart_coords for mol in molecules]
    site_properties = [mol.site_properties for mol in molecules]
    return cls(species=species, coords=coords, charge=int(molecules[0].charge), spin_multiplicity=int(molecules[0].spin_multiplicity), site_properties=site_properties, **kwargs)