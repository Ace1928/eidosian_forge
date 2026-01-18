import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_default_handler_indirect(self):

    def default(obj):
        if isinstance(obj, complex):
            return [('mathjs', 'Complex'), ('re', obj.real), ('im', obj.imag)]
        return str(obj)
    df_list = [9, DataFrame({'a': [1, 'STR', complex(4, -5)], 'b': [float('nan'), None, 'N/A']}, columns=['a', 'b'])]
    expected = '[9,[[1,null],["STR",null],[[["mathjs","Complex"],["re",4.0],["im",-5.0]],"N\\/A"]]]'
    assert ujson_dumps(df_list, default_handler=default, orient='values') == expected