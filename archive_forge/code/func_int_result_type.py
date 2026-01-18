import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def int_result_type(dtype, dtype2):
    typs = {dtype.kind, dtype2.kind}
    if not len(typs - {'i', 'u', 'b'}) and (dtype.kind == 'i' or dtype2.kind == 'i'):
        return 'i'
    elif not len(typs - {'u', 'b'}) and (dtype.kind == 'u' or dtype2.kind == 'u'):
        return 'u'
    return None