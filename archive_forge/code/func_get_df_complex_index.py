import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def get_df_complex_index():
    df = get_countries()
    df = df.reset_index().set_index(['region', 'country'])
    df.columns = pd.MultiIndex.from_arrays([['code' if col == 'code' else 'localisation' if col in ['longitude', 'latitude'] else 'data' for col in df.columns], df.columns], names=['category', 'detail'])
    return df