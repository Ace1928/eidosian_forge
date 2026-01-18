from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
def set_gw_options(self, nv_band=10, nc_band=10, n_iteration=5, n_grid=6, dE_grid=0.5):
    """
        Set parameters in cell.in for a GW computation

        Args:
            nv__band: number of valence bands to correct with GW
            nc_band: number of conduction bands to correct with GW
            n_iteration: number of iteration
            n_grid and dE_grid: number of points and spacing in eV for correlation grid.
        """
    self.GW_options.update(nv_corr=nv_band, nc_corr=nc_band, nit_gw=n_iteration)
    self.correlation_grid.update(dE_grid=dE_grid, n_grid=n_grid)