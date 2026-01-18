import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
def download_chunk(*, range_start, range_end, headers, download_path, http_uri):
    combined_headers = {**headers, 'Range': f'bytes={range_start}-{range_end}'}
    with cloud_storage_http_request('get', http_uri, stream=False, headers=combined_headers, timeout=10) as response:
        expected_length = response.headers.get('Content-Length')
        if expected_length is not None:
            actual_length = response.raw.tell()
            expected_length = int(expected_length)
            if actual_length < expected_length:
                raise IOError('Incomplete read ({} bytes read, {} more expected)'.format(actual_length, expected_length - actual_length))
        augmented_raise_for_status(response)
        with open(download_path, 'r+b') as f:
            f.seek(range_start)
            f.write(response.content)