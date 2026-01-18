import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def generate_random_df(rows, columns, column_types=COLUMN_TYPES):
    rows = int(rows)
    types = np.random.choice(column_types, size=columns)
    columns = ['Column{}OfType{}'.format(col, type.title()) for col, type in enumerate(types)]
    series = {col: generate_random_series(rows, type) for col, type in zip(columns, types)}
    index = pd.Index(range(rows))
    for x in series.values():
        x.index = index
    return pd.DataFrame(series)