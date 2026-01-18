import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
def cloud_storage_http_request(method, url, max_retries=5, backoff_factor=2, backoff_jitter=1.0, retry_codes=_TRANSIENT_FAILURE_RESPONSE_CODES, timeout=None, **kwargs):
    """Performs an HTTP PUT/GET/PATCH request using Python's `requests` module with automatic retry.

    Args:
        method: string of 'PUT' or 'GET' or 'PATCH', specify to do http PUT or GET or PATCH.
        url: the target URL address for the HTTP request.
        max_retries: maximum number of retries before throwing an exception.
        backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
            request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
            exponential backoff.
        backoff_jitter: A random jitter to add to the backoff interval.
        retry_codes: a list of HTTP response error codes that qualifies for retry.
        timeout: wait for timeout seconds for response from remote server for connect and
            read request. Default to None owing to long duration operation in read / write.
        kwargs: Additional keyword arguments to pass to `requests.Session.request()`.

    Returns:
        requests.Response object.
    """
    if method.lower() not in ('put', 'get', 'patch', 'delete'):
        raise ValueError('Illegal http method: ' + method)
    return _get_http_response_with_retries(method, url, max_retries, backoff_factor, backoff_jitter, retry_codes, timeout=timeout, **kwargs)