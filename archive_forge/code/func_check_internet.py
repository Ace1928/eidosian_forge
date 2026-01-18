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
def check_internet(url=None):
    """Check if internet is available"""
    url = 'https://github.com' if url is None else url
    try:
        urlopen(url)
    except URLError as err:
        return False
    return True