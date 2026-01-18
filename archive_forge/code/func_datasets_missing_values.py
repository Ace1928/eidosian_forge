import gzip
import json
import os
import re
from functools import partial
from importlib import resources
from io import BytesIO
from urllib.error import HTTPError
import numpy as np
import pytest
import scipy.sparse
import sklearn
from sklearn import config_context
from sklearn.datasets import fetch_openml as fetch_openml_orig
from sklearn.datasets._openml import (
from sklearn.utils import Bunch, check_pandas_support
from sklearn.utils._testing import (
@pytest.fixture(scope='module')
def datasets_missing_values():
    return {61: {}, 2: {'family': 11, 'temper_rolling': 9, 'condition': 2, 'formability': 4, 'non-ageing': 10, 'surface-finish': 11, 'enamelability': 11, 'bc': 11, 'bf': 10, 'bt': 11, 'bw%2Fme': 8, 'bl': 9, 'm': 11, 'chrom': 11, 'phos': 11, 'cbond': 10, 'marvi': 11, 'exptl': 11, 'ferro': 11, 'corr': 11, 'blue%2Fbright%2Fvarn%2Fclean': 11, 'lustre': 8, 'jurofm': 11, 's': 11, 'p': 11, 'oil': 10, 'packing': 11}, 561: {}, 40589: {}, 1119: {}, 40966: {'BCL2_N': 7}, 40945: {'age': 263, 'fare': 1, 'cabin': 1014, 'embarked': 2, 'boat': 823, 'body': 1188, 'home.dest': 564}}