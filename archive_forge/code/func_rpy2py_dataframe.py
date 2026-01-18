import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import na_values
from rpy2.rinterface import IntSexpVector
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface import SexpVector
from rpy2.rinterface import StrSexpVector
import datetime
import functools
import math
import numpy  # type: ignore
import pandas  # type: ignore
import pandas.core.series  # type: ignore
from pandas.core.frame import DataFrame as PandasDataFrame  # type: ignore
from pandas.core.dtypes.api import is_datetime64_any_dtype  # type: ignore
import warnings
from collections import OrderedDict
from rpy2.robjects.vectors import (BoolVector,
import rpy2.robjects.numpy2ri as numpy2ri
@rpy2py.register(DataFrame)
def rpy2py_dataframe(obj):
    rpy2py = conversion.get_conversion().rpy2py
    colnames_lst = []
    od = OrderedDict(((i, rpy2py(col) if isinstance(col, rinterface.SexpVector) else col) for i, col in enumerate(_flatten_dataframe(obj, colnames_lst))))
    res = pandas.DataFrame.from_dict(od)
    res.columns = tuple(('.'.join(_) if isinstance(_, list) else _ for _ in colnames_lst))
    res.index = obj.rownames
    return res