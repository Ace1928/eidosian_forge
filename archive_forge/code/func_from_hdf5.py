from __future__ import annotations
import itertools
import json
import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from scipy.interpolate import RegularGridInterpolator
from pymatgen.core import Element, Site, Structure
from pymatgen.core.units import ang_to_bohr, bohr_to_angstrom
from pymatgen.electronic_structure.core import Spin
@classmethod
def from_hdf5(cls, filename: str, **kwargs) -> Self:
    """
        Reads VolumetricData from HDF5 file.

        Args:
            filename: Filename

        Returns:
            VolumetricData
        """
    import h5py
    with h5py.File(filename, mode='r') as file:
        data = {k: np.array(v) for k, v in file['vdata'].items()}
        data_aug = None
        if 'vdata_aug' in file:
            data_aug = {k: np.array(v) for k, v in file['vdata_aug'].items()}
        structure = Structure.from_dict(json.loads(file.attrs['structure_json']))
        return cls(structure, data=data, data_aug=data_aug, **kwargs)