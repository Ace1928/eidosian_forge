from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def env_body(self) -> str:
    iterator = self._create_row_iterator(over='body')
    return '\n'.join(list(iterator))