import pytest
import os
from ase.db import connect
def test_delete_vacuum():
    db = connect(db_name)
    write_entries_to_db(db)
    check_delete_function(db)