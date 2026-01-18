import pytest
from ase.io import read, write
from ase.build import molecule
from ase.test.factories import ObsoleteFactoryWrapper
@calc('gpaw', mode='lcao', basis='sz(dzp)', marks=pytest.mark.filterwarnings('ignore:The keyword'))
@calc('abinit', 'cp2k', 'emt')
@calc('vasp', xc='lda', prec='low')
def test_h2_traj(factory, testdir):
    run(factory)