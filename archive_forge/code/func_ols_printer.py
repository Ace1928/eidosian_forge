from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def ols_printer():
    """
        print summary table for ols models
        """
    table = str(general_table) + '\n' + str(parameter_table)
    return table