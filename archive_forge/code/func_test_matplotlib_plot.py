import os
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms
def test_matplotlib_plot(plt):
    slab = FaceCenteredCubic('Au', size=(2, 2, 2))
    fig, ax = plt.subplots()
    plot_atoms(slab, ax, radii=0.5, rotation='10x,10y,10z', show_unit_cell=0)
    assert len(ax.patches) == len(slab)