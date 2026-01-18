from statsmodels.compat.python import lrange
from io import StringIO
from os import environ, makedirs
from os.path import abspath, dirname, exists, expanduser, join
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen
import numpy as np
from pandas import Index, read_csv, read_stata
def process_pandas(data, endog_idx=0, exog_idx=None, index_idx=None):
    names = data.columns
    if isinstance(endog_idx, int):
        endog_name = names[endog_idx]
        endog = data[endog_name].copy()
        if exog_idx is None:
            exog = data.drop([endog_name], axis=1)
        else:
            exog = data[names[exog_idx]].copy()
    else:
        endog = data.loc[:, endog_idx].copy()
        endog_name = list(endog.columns)
        if exog_idx is None:
            exog = data.drop(endog_name, axis=1)
        elif isinstance(exog_idx, int):
            exog = data[names[exog_idx]].copy()
        else:
            exog = data[names[exog_idx]].copy()
    if index_idx is not None:
        index = Index(data.iloc[:, index_idx])
        endog.index = index
        exog.index = index.copy()
        data = data.set_index(names[index_idx])
    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=list(names), endog=endog, exog=exog, endog_name=endog_name, exog_name=exog_name)
    return dataset