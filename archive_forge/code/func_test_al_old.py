import pytest
from ase.build import bulk
from ase.test.factories import ObsoleteFactoryWrapper
@pytest.mark.parametrize('name', sorted(required))
def test_al_old(name):
    factory = ObsoleteFactoryWrapper(name)
    run(factory)