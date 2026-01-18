import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
def get_offset_string(self, offset, limit=None):
    return '\nOFFSET {0}'.format(offset)