from pathlib import Path
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.db import connect
def test_analyse(cli, dbfile):
    txt = cli.ase('db', dbfile, '--show-keys')
    print(txt)
    assert 'carrots: 2' in txt
    assert 'oranges: 1' in txt