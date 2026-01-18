from __future__ import annotations
from collections.abc import (
import csv as csvlib
import os
from typing import (
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
@property
def encoded_labels(self) -> list[Hashable]:
    encoded_labels: list[Hashable] = []
    if self.index and self.index_label:
        assert isinstance(self.index_label, Sequence)
        encoded_labels = list(self.index_label)
    if not self.has_mi_columns or self._has_aliases:
        encoded_labels += list(self.write_cols)
    return encoded_labels