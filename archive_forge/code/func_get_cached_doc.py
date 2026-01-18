from typing import Optional
import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
def get_cached_doc(cached_prop) -> Optional[str]:
    return get_cached_func(cached_prop).__doc__