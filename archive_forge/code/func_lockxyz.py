from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
@lockxyz.setter
def lockxyz(self, lockxyz):
    self.structure.add_site_property('selective_dynamics', lockxyz)