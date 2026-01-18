import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.spectrum.band_structure import calculate_band_structure
from ase.calculators.test import FreeElectrons
from ase.cell import Cell
def test_bandstructure_transform_mcl(testdir):

    def _atoms(cell):
        atoms = Atoms(cell=cell, pbc=True)
        atoms.calc = FreeElectrons()
        return atoms
    cell = Cell.new([3.0, 5.0, 4.0, 90.0, 110.0, 90.0])
    lat = cell.get_bravais_lattice()
    density = 10.0
    cell0 = lat.tocell()
    path0 = lat.bandpath(density=density)
    print(cell.cellpar().round(3))
    print(cell0.cellpar().round(3))
    with workdir('files', mkdir=True):
        bs = calculate_band_structure(_atoms(cell), cell.bandpath(density=density))
        bs.write('bs.json')
        bs0 = calculate_band_structure(_atoms(cell0), path0)
        bs0.write('bs0.json')
    maxerr = np.abs(bs.energies - bs0.energies).max()
    assert maxerr < 1e-12, maxerr