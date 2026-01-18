from pathlib import Path
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.db import connect
def test_insert_into(cli, dbfile):
    """Test --insert-into."""
    out = Path(dbfile).with_name('x1.db')
    cli.ase(*f'db {dbfile} --limit 1 --insert-into {out} --progress-bar'.split())
    txt = cli.ase(*f'db {out} --count'.split())
    num = int(txt.split()[0])
    assert num == 1