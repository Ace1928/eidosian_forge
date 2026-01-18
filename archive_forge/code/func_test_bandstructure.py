import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.test import FreeElectrons
from ase.dft.kpoints import special_paths
from ase.spectrum.band_structure import BandStructure
def test_bandstructure(testdir, plt):
    atoms = bulk('Cu')
    path = special_paths['fcc']
    atoms.calc = FreeElectrons(nvalence=1, kpts={'path': path, 'npoints': 200})
    atoms.get_potential_energy()
    bs = atoms.calc.band_structure()
    coords, labelcoords, labels = bs.get_labels()
    print(labels)
    bs.write('hmm.json')
    bs = BandStructure.read('hmm.json')
    coords, labelcoords, labels = bs.get_labels()
    print(labels)
    assert ''.join(labels) == 'GXWKGLUWLKUX'
    bs.plot(emax=10, filename='bs.png')