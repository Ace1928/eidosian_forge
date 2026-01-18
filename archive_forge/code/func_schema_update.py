import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
def schema_update(sql):
    for a, b in [('REAL', 'DOUBLE PRECISION'), ('INTEGER PRIMARY KEY AUTOINCREMENT', 'SERIAL PRIMARY KEY')]:
        sql = sql.replace(a, b)
    arrays_1D = ['numbers', 'initial_magmoms', 'initial_charges', 'masses', 'tags', 'momenta', 'stress', 'dipole', 'magmoms', 'charges']
    arrays_2D = ['positions', 'cell', 'forces']
    txt2jsonb = ['calculator_parameters', 'key_value_pairs']
    for column in arrays_1D:
        if column in ['numbers', 'tags']:
            dtype = 'INTEGER'
        else:
            dtype = 'DOUBLE PRECISION'
        sql = sql.replace('{} BLOB,'.format(column), '{} {}[],'.format(column, dtype))
    for column in arrays_2D:
        sql = sql.replace('{} BLOB,'.format(column), '{} DOUBLE PRECISION[][],'.format(column))
    for column in txt2jsonb:
        sql = sql.replace('{} TEXT,'.format(column), '{} JSONB,'.format(column))
    sql = sql.replace('data BLOB,', 'data JSONB,')
    return sql