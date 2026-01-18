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
def start_processing_headers(self):
    """
        shared logic at the start of a GET request
        """
    self.send_response(200)
    self.requested_from_user_agent = self.headers['User-Agent']
    response_df = pd.DataFrame({'header': [self.requested_from_user_agent]})
    return response_df