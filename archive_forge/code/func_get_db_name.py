import os
import pytest
from ase.db import connect
from ase import Atoms
import numpy as np
def get_db_name(name):
    if name == 'postgresql':
        if os.environ.get('POSTGRES_DB'):
            name = 'postgresql://ase:ase@postgres:5432/testase'
        else:
            name = os.environ.get('ASE_TEST_POSTGRES_URL')
    elif name == 'mysql':
        if os.environ.get('CI_PROJECT_DIR'):
            name = 'mysql://root:ase@mysql:3306/testase_mysql'
        else:
            name = os.environ.get('MYSQL_DB_URL')
    elif name == 'mariadb':
        if os.environ.get('CI_PROJECT_DIR'):
            name = 'mariadb://root:ase@mariadb:3306/testase_mysql'
        else:
            name = os.environ.get('MYSQL_DB_URL')
    return name