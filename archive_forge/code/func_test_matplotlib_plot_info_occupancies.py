import os
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms
def test_matplotlib_plot_info_occupancies(plt):
    slab = FaceCenteredCubic('Au')
    slab.info['occupancy'] = {'0': {'Au': 1}}
    fig, ax = plt.subplots()
    plot_atoms(slab, ax, show_unit_cell=0)
    assert len(ax.patches) == len(slab)