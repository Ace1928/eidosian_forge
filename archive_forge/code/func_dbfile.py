from pathlib import Path
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.db import connect
@pytest.fixture(scope='module')
def dbfile(tmp_path_factory) -> str:
    """Create a database file (x.db) with two rows."""
    path = tmp_path_factory.mktemp('db') / 'x.db'
    with connect(path) as db:
        db.write(Atoms())
        db.write(molecule('H2O'), key_value_pairs={'carrots': 3})
        db.write(bulk('Ti'), key_value_pairs={'oranges': 42, 'carrots': 4})
    return str(path)