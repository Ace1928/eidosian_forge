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
def strip_column_names(df):
    """
    Remove leading and trailing single quotes

    Parameters
    ----------
    df : DataFrame
        DataFrame to process

    Returns
    -------
    df : DataFrame
        DataFrame with stripped column names

    Notes
    -----
    In-place modification
    """
    columns = []
    for c in df:
        if c.startswith("'") and c.endswith("'"):
            c = c[1:-1]
        elif c.startswith("'"):
            c = c[1:]
        elif c.endswith("'"):
            c = c[:-1]
        columns.append(c)
    df.columns = columns
    return df