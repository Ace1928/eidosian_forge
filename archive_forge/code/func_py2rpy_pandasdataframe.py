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
@py2rpy.register(PandasDataFrame)
def py2rpy_pandasdataframe(obj):
    if obj.index.duplicated().any():
        warnings.warn('DataFrame contains duplicated elements in the index, which will lead to loss of the row names in the resulting data.frame')
    od = OrderedDict()
    for name, values in obj.items():
        try:
            od[name] = conversion.converter_ctx.get().py2rpy(values)
        except Exception as e:
            warnings.warn('Error while trying to convert the column "%s". Fall back to string conversion. The error is: %s' % (name, str(e)))
            od[name] = conversion.converter_ctx.get().py2rpy(values.astype('string'))
    return DataFrame(od)