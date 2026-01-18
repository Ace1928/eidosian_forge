import logging
from typing import Dict, Optional
import xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
def num_row(self):
    """
        Get number of rows.

        Returns
        -------
        int
        """
    return self._n_rows