import unittest
from unittest import SkipTest
import numpy as np
import pandas as pd
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.util import pandas_version
from holoviews.util.transform import dim
from .test_pandasinterface import BasePandasInterfaceTests
def test_dataset_range_categorical_dimension(self):
    ddf = dd.from_pandas(pd.DataFrame({'a': ['1', '2', '3']}), 1)
    ds = Dataset(ddf)
    self.assertEqual(ds.range(0), ('1', '3'))