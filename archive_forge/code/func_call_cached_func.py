from typing import Optional
import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
def call_cached_func(cached_prop, *args, **kwargs):
    f = get_cached_func(cached_prop)
    return f(*args, **kwargs)