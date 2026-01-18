import pytest
from ase.build import bulk
from ase.test.factories import ObsoleteFactoryWrapper
@calc('abinit', ecut=200, toldfe=0.0001, chksymbreak=0)
@calc('openmx', **omx_par)
@calc('elk', rgkmax=5.0)
def test_al(factory):
    run(factory)