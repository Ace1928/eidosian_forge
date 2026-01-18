from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def set_structures(self, structures: Sequence[Structure], tags=None):
    """
        Add list of structures to the visualizer.

        Args:
            structures (list[Structures]): structures to be visualized.
            tags (): List of tags.
        """
    self.structures = structures
    self.istruct = 0
    self.current_structure = self.structures[self.istruct]
    self.tags = tags if tags is not None else []
    self.all_radii = []
    self.all_vis_radii = []
    for struct in self.structures:
        struct_radii = []
        struct_vis_radii = []
        for site in struct:
            radius = 0
            for specie, occu in site.species.items():
                radius += occu * (specie.ionic_radius if isinstance(specie, Species) and specie.ionic_radius else specie.average_ionic_radius)
                vis_radius = 0.2 + 0.002 * radius
            struct_radii.append(radius)
            struct_vis_radii.append(vis_radius)
        self.all_radii.append(struct_radii)
        self.all_vis_radii.append(struct_vis_radii)
    self.set_structure(self.current_structure, reset_camera=True, to_unit_cell=False)