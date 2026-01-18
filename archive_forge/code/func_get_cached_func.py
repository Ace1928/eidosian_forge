from typing import Optional
import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
def get_cached_func(cached_prop):
    try:
        return cached_prop.fget
    except AttributeError:
        return cached_prop.func