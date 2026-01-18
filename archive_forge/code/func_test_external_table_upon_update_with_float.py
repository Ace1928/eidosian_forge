import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def test_external_table_upon_update_with_float(db_name):
    db = connect(db_name)
    ext_table = {'value1': 1.0, 'value2': 2.0}
    atoms = Atoms('Pb', positions=[[0, 0, 0]])
    uid = db.write(atoms)
    db.update(uid, external_tables={'float_table': ext_table})