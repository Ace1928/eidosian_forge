from __future__ import print_function
import os
import io
import time
import functools
import collections
import collections.abc
import numpy as np
import requests
import IPython
def download_yield_bytes(url, chunk_size=1024 * 1024 * 10):
    """Yield a downloaded url as byte chunks.

    :param url: str or url
    :param chunk_size: None or int in bytes
    :yield: byte chunks
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_length = response.headers.get('content-length')
    if total_length is not None:
        total_length = float(total_length)
        length_str = '{0:.2f}Mb '.format(total_length / (1024 * 1024))
    else:
        length_str = ''
    print('Yielding {0:s} {1:s}'.format(url, length_str))
    for chunk in response.iter_content(chunk_size=chunk_size):
        yield chunk
    response.close()