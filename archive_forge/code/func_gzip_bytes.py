import gzip
import http.server
from io import BytesIO
import multiprocessing
import socket
import time
import urllib.error
import pytest
from pandas.compat import is_ci_environment
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
def gzip_bytes(self, response_bytes):
    """
        some web servers will send back gzipped files to save bandwidth
        """
    with BytesIO() as bio:
        with gzip.GzipFile(fileobj=bio, mode='w') as zipper:
            zipper.write(response_bytes)
        response_bytes = bio.getvalue()
    return response_bytes