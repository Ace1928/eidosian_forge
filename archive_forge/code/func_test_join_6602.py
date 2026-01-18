import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_join_6602():
    abbreviations = pd.Series(['Major League Baseball', 'National Basketball Association'], index=['MLB', 'NBA'])
    teams = pd.DataFrame({'name': ['Mariners', 'Lakers'] * 50, 'league_abbreviation': ['MLB', 'NBA'] * 50})
    with warnings.catch_warnings():
        warnings.filterwarnings('error', "Distributing <class 'dict'> object", category=UserWarning)
        teams.set_index('league_abbreviation').join(abbreviations.rename('league_name'))