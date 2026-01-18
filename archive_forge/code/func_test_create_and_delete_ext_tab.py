import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def test_create_and_delete_ext_tab(db_name):
    ext_tab = ['tab1', 'tab2', 'tab3']
    atoms = Atoms()
    db = connect(name)
    db.write(atoms)
    for tab in ext_tab:
        db._create_table_if_not_exists(tab, 'INTEGER')
    current_ext_tables = db._get_external_table_names()
    for tab in ext_tab:
        assert tab in current_ext_tables
    db.delete_external_table('tab1')
    assert 'tab1' not in db._get_external_table_names()