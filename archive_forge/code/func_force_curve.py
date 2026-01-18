import numpy as np
from collections import namedtuple
from ase.geometry import find_mic
def force_curve(images, ax=None):
    """Plot energies and forces as a function of accumulated displacements.

    This is for testing whether a calculator's forces are consistent with
    the energies on a set of geometries where energies and forces are
    available."""
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    nim = len(images)
    accumulated_distances = []
    accumulated_distance = 0.0
    energies = [atoms.get_potential_energy() for atoms in images]
    for i in range(nim):
        atoms = images[i]
        f_ac = atoms.get_forces()
        if i < nim - 1:
            rightpos = images[i + 1].positions
        else:
            rightpos = atoms.positions
        if i > 0:
            leftpos = images[i - 1].positions
        else:
            leftpos = atoms.positions
        disp_ac, _ = find_mic(rightpos - leftpos, cell=atoms.cell, pbc=atoms.pbc)

        def total_displacement(disp):
            disp_a = (disp ** 2).sum(axis=1) ** 0.5
            return sum(disp_a)
        dE_fdotr = -0.5 * np.vdot(f_ac.ravel(), disp_ac.ravel())
        linescale = 0.45
        disp = 0.5 * total_displacement(disp_ac)
        if i == 0 or i == nim - 1:
            disp *= 2
            dE_fdotr *= 2
        x1 = accumulated_distance - disp * linescale
        x2 = accumulated_distance + disp * linescale
        y1 = energies[i] - dE_fdotr * linescale
        y2 = energies[i] + dE_fdotr * linescale
        ax.plot([x1, x2], [y1, y2], 'b-')
        ax.plot(accumulated_distance, energies[i], 'bo')
        ax.set_ylabel('Energy [eV]')
        ax.set_xlabel('Accumulative distance [Å]')
        accumulated_distances.append(accumulated_distance)
        accumulated_distance += total_displacement(rightpos - atoms.positions)
    ax.plot(accumulated_distances, energies, ':', zorder=-1, color='k')
    return ax