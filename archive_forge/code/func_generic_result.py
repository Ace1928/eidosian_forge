from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def generic_result(self):
    if self.cmp_result is None:
        return NotImplemented
    else:
        return self.cmp_result