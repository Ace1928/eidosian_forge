import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
def get_last_id(self, cur):
    cur.execute('SELECT last_value FROM systems_id_seq')
    id = cur.fetchone()[0]
    return int(id)