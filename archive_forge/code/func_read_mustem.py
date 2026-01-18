import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer
from .utils import verify_cell_for_export, verify_dictionary
@reader
def read_mustem(fd):
    """Import muSTEM input file.

    Reads cell, atom positions, etc. from muSTEM xtl file.
    The mustem xtl save the root mean square (RMS) displacement, which is
    convert to Debye-Waller (in Å²) factor by:

    .. math::

        B = RMS * 8\\pi^2

    """
    from ase.geometry import cellpar_to_cell
    fd.readline()
    cellpar = [float(i) for i in fd.readline().split()[:3]]
    cell = cellpar_to_cell(cellpar)
    fd.readline()
    element_number = int(fd.readline().strip())
    atomic_numbers = []
    positions = []
    debye_waller_factors = []
    occupancies = []
    for i in range(element_number):
        _ = fd.readline()
        line = fd.readline().split()
        atoms_number = int(line[0])
        atomic_number = int(line[1])
        occupancy = float(line[2])
        DW = float(line[3]) * 8 * np.pi ** 2
        positions.append(np.genfromtxt(fname=fd, max_rows=atoms_number))
        atomic_numbers.append(np.ones(atoms_number, dtype=int) * atomic_number)
        occupancies.append(np.ones(atoms_number) * occupancy)
        debye_waller_factors.append(np.ones(atoms_number) * DW)
    positions = np.vstack(positions)
    atoms = Atoms(cell=cell, scaled_positions=positions)
    atoms.set_atomic_numbers(np.hstack(atomic_numbers))
    atoms.set_array('occupancies', np.hstack(occupancies))
    atoms.set_array('debye_waller_factors', np.hstack(debye_waller_factors))
    return atoms