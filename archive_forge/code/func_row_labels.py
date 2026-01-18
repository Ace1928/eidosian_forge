import numpy as np
import pandas as pd
from scipy import stats
@property
def row_labels(self):
    """The row labels used in pandas-types."""
    return self._row_labels