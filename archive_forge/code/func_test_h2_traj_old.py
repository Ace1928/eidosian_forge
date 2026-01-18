import pytest
from ase.io import read, write
from ase.build import molecule
from ase.test.factories import ObsoleteFactoryWrapper
@pytest.mark.parametrize('name', sorted(parameters))
def test_h2_traj_old(name, testdir):
    factory = ObsoleteFactoryWrapper(name)
    run(factory)