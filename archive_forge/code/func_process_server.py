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
def process_server(responder, port):
    with http.server.HTTPServer(('localhost', port), responder) as server:
        server.handle_request()
    server.server_close()