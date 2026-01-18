import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
def deblob(self, buf, dtype=float, shape=None):
    """Convert blob/buffer object to ndarray of correct dtype and shape.

        (without creating an extra view)."""
    if buf is None:
        return None
    return np.array(buf, dtype=dtype)