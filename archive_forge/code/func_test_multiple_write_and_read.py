import filecmp
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
def test_multiple_write_and_read():
    images = []
    for name in ['C6H6', 'H2O', 'CO']:
        images.append(molecule(name))
    write('1.xyz', images, format='extxyz', plain=True)
    write('2.xyz', images, format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'
    images1 = read('1.xyz', format='xyz', index=':')
    assert len(images) == len(images1)
    for atoms, atoms1 in zip(images, images1):
        assert atoms_equal(atoms, atoms1), 'Read failed'