import numbers
from functools import reduce
from operator import mul
import numpy as np
def shrink_data(self):
    self._data.resize((self._get_next_offset(),) + self.common_shape, refcheck=False)