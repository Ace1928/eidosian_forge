import math
from collections import OrderedDict
from datetime import datetime
import pytest
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import vectors
from rpy2.robjects import conversion
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
def test_dataframe_warns_duplicated_index(self):
    pd_df = pandas.DataFrame({'x': [1, 2]}, index=['a', 'a'])
    with pytest.warns(UserWarning, match='DataFrame contains duplicated elements in the index') as record:
        with localconverter(default_converter + rpyp.converter) as cv:
            robjects.conversion.converter_ctx.get().py2rpy(pd_df)
    assert len(record) == 1