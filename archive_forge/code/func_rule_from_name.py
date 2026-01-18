from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
def rule_from_name(self, name: str):
    for rule in self.rules:
        if rule.name == name:
            return rule
    return None