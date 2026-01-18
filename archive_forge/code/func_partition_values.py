import sys
import numpy as np
from .dask import DaskInterface
from .interface import Interface
from .spatialpandas import SpatialPandasInterface
@classmethod
def partition_values(cls, df, dataset, dimension, expanded, flat):
    ds = dataset.clone(df, datatype=['spatialpandas'])
    return ds.interface.values(ds, dimension, expanded, flat)