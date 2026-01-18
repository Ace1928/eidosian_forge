import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
def test_doc_module():
    import pandas
    import modin.pandas as pd
    from modin.config import DocModule
    DocModule.put('modin.tests.config.docs_module')
    assert pd.DataFrame.apply.__doc__ == 'This is a test of the documentation module for DataFrame.'
    assert pandas.DataFrame.isna.__doc__ in pd.DataFrame.isna.__doc__
    assert pandas.DataFrame.isnull.__doc__ in pd.DataFrame.isnull.__doc__
    assert pd.Series.isna.__doc__ == 'This is a test of the documentation module for Series.'
    assert pandas.Series.isnull.__doc__ in pd.Series.isnull.__doc__
    assert pandas.Series.apply.__doc__ in pd.Series.apply.__doc__
    assert pd.read_csv.__doc__ == 'Test override for functions on the module.'
    assert pandas.read_table.__doc__ in pd.read_table.__doc__