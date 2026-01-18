from contextlib import nullcontext
from datetime import (
import locale
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def get_local_am_pm():
    """Return the AM and PM strings returned by strftime in current locale."""
    am_local = time(1).strftime('%p')
    pm_local = time(13).strftime('%p')
    return (am_local, pm_local)