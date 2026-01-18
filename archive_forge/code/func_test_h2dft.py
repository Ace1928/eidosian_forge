import pytest
from ase.build import molecule
from ase.test.factories import ObsoleteFactoryWrapper
@calc('cp2k', auto_write=True, uks=True)
def test_h2dft(factory):
    run(factory)