import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.calculators.calculator import PropertyNotImplementedError
def plot_with_colors(self, ax=None, emin=-10, emax=5, filename=None, show=False, energies=None, colors=None, ylabel=None, clabel='$s_z$', cmin=-1.0, cmax=1.0, sortcolors=False, loc=None, s=2):
    """Plot band-structure with colors."""
    import matplotlib.pyplot as plt
    if self.ax is None:
        ax = self.prepare_plot(ax, emin, emax, ylabel)
    shape = energies.shape
    xcoords = np.vstack([self.xcoords] * shape[1])
    if sortcolors:
        perm = colors.argsort(axis=None)
        energies = energies.ravel()[perm].reshape(shape)
        colors = colors.ravel()[perm].reshape(shape)
        xcoords = xcoords.ravel()[perm].reshape(shape)
    for e_k, c_k, x_k in zip(energies, colors, xcoords):
        things = ax.scatter(x_k, e_k, c=c_k, s=s, vmin=cmin, vmax=cmax)
    cbar = plt.colorbar(things)
    cbar.set_label(clabel)
    self.finish_plot(filename, show, loc)
    return ax