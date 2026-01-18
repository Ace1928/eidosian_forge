import sys
import numpy as np
import pandas as pd
from .. import util
from ..dimension import Dimension
from ..element import Element
from ..ndmapping import NdMapping, item_check, sorted_context
from .interface import Interface
from .pandas import PandasInterface
@classmethod
def concat_fn(cls, dataframes, **kwargs):
    import dask.dataframe as dd
    return dd.concat(dataframes, **kwargs)