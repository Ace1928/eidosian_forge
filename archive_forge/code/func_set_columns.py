from typing import Optional, List
import numpy as np
from ase.db.core import float_to_time_string, now
def set_columns(self, columns):
    self.values = []
    for c in columns:
        if c == 'age':
            value = float_to_time_string(now() - self.dct.ctime)
        elif c == 'pbc':
            value = ''.join(('FT'[int(p)] for p in self.dct.pbc))
        else:
            value = getattr(self.dct, c, None)
        self.values.append(value)