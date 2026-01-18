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
def download_to_bytes(url, chunk_size=1024 * 1024 * 10, loadbar_length=10):
    """Download a url to bytes.

    if chunk_size is not None, prints a simple loading bar [=*loadbar_length] to show progress (in console and notebook)

    :param url: str or url
    :param chunk_size: None or int in bytes
    :param loadbar_length: int length of load bar
    :return: (bytes, encoding)
    """
    stream = False if chunk_size is None else True
    print('Downloading {0:s}: '.format(url), end='')
    response = requests.get(url, stream=stream)
    response.raise_for_status()
    encoding = response.encoding
    total_length = response.headers.get('content-length')
    if total_length is not None:
        total_length = float(total_length)
        if stream:
            print('{0:.2f}Mb/{1:} '.format(total_length / (1024 * 1024), loadbar_length), end='')
        else:
            print('{0:.2f}Mb '.format(total_length / (1024 * 1024)), end='')
    if stream:
        print('[', end='')
        chunks = []
        loaded = 0
        loaded_size = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                if total_length is not None:
                    while loaded < loadbar_length * loaded_size / total_length:
                        print('=', end='')
                        loaded += 1
                    loaded_size += chunk_size
                chunks.append(chunk)
        if total_length is None:
            print('=' * loadbar_length, end='')
        else:
            while loaded < loadbar_length:
                print('=', end='')
                loaded += 1
        content = b''.join(chunks)
        print('] ', end='')
    else:
        content = response.content
    print('Finished')
    response.close()
    return (content, encoding)