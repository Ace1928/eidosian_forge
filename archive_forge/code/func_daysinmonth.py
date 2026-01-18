from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@property
def daysinmonth(self):
    return self._Series(query_compiler=self._query_compiler.dt_daysinmonth())