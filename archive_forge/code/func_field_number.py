import collections.abc
import copy
import pickle
from typing import (
@property
def field_number(self):
    self._check_valid()
    return self._parent._internal_get(self._index)._field_number