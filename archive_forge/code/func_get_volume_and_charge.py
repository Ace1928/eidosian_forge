from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
def get_volume_and_charge(nonequiv_idx):
    attractor = yt['integration']['attractors'][nonequiv_idx - 1]
    if attractor['id'] != nonequiv_idx:
        raise ValueError(f'List of attractors may be un-ordered (wanted id={nonequiv_idx}): {attractor}')
    return (attractor['integrals'][volume_idx], attractor['integrals'][charge_idx])