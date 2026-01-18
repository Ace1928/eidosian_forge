import math
import string
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import cycle
import numpy as np
import pandas as pd
from .utils import find_package_file
def get_population():
    """A Pandas Series with the world population (from the world bank data)"""
    return pd.read_csv(find_package_file('samples/population.csv')).set_index('Country')['SP.POP.TOTL']