from __future__ import annotations
import gzip
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.core import Lattice, Molecule, Structure
def get_species_block(self, structure: Structure | Molecule, species_dir: str | Path) -> str:
    """Get the basis set information for a structure

        Args:
            structure (Molecule or Structure): The structure to get the basis set information for
            species_dir (str or Pat:): The directory to find the species files in

        Returns:
            The block to add to the control.in file for the species

        Raises:
            ValueError: If a file for the species is not found
        """
    block = ''
    species = np.unique(structure.species)
    for sp in species:
        filename = f'{species_dir}/{sp.Z:02d}_{sp.symbol}_default'
        if Path(filename).exists():
            with open(filename) as sf:
                block += ''.join(sf.readlines())
        elif Path(f'{filename}.gz').exists():
            with gzip.open(f'{filename}.gz', mode='rt') as sf:
                block += ''.join(sf.readlines())
        else:
            raise ValueError(f'Species file for {sp.symbol} not found.')
    return block