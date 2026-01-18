import pytest
from ase import Atoms
from ase.db import connect
from ase.db.web import Session
@pytest.fixture(scope='module')
def database(tmp_path_factory):
    with tmp_path_factory.mktemp('dbtest') as dbtest:
        db = connect(dbtest / 'test.db', append=False)
        x = [0, 1, 2]
        t1 = [1, 2, 0]
        t2 = [[2, 3], [1, 1], [1, 0]]
        atoms = Atoms('H2O', [(0, 0, 0), (2, 0, 0), (1, 1, 0)])
        atoms.center(vacuum=5)
        atoms.set_pbc(True)
        db.write(atoms, foo=42.0, bar='abc', data={'x': x, 't1': t1, 't2': t2})
        db.write(atoms)
        yield db