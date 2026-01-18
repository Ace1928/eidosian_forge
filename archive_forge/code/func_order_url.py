import os
from unittest import mock
import fixtures
import requests
import testtools
def order_url(url):
    """Returns the url with the query strings ordered, if they exist and
    there's more than one. Otherwise the url is returned unaltered.
    """
    if URL_QUERY_SEPARATOR in url:
        parts = url.split(URL_SEPARATOR)
        if len(parts) == 2:
            queries = sorted(parts[1].split(URL_QUERY_SEPARATOR))
            url = URL_SEPARATOR.join([parts[0], URL_QUERY_SEPARATOR.join(queries)])
    return url