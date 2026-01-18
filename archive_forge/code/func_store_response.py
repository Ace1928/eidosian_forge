import gzip
import logging
import pickle
from email.utils import mktime_tz, parsedate_tz
from importlib import import_module
from pathlib import Path
from time import time
from weakref import WeakKeyDictionary
from w3lib.http import headers_dict_to_raw, headers_raw_to_dict
from scrapy.http import Headers, Response
from scrapy.http.request import Request
from scrapy.responsetypes import responsetypes
from scrapy.spiders import Spider
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.project import data_path
from scrapy.utils.python import to_bytes, to_unicode
def store_response(self, spider: Spider, request: Request, response):
    """Store the given response in the cache."""
    rpath = Path(self._get_request_path(spider, request))
    if not rpath.exists():
        rpath.mkdir(parents=True)
    metadata = {'url': request.url, 'method': request.method, 'status': response.status, 'response_url': response.url, 'timestamp': time()}
    with self._open(rpath / 'meta', 'wb') as f:
        f.write(to_bytes(repr(metadata)))
    with self._open(rpath / 'pickled_meta', 'wb') as f:
        pickle.dump(metadata, f, protocol=4)
    with self._open(rpath / 'response_headers', 'wb') as f:
        f.write(headers_dict_to_raw(response.headers))
    with self._open(rpath / 'response_body', 'wb') as f:
        f.write(response.body)
    with self._open(rpath / 'request_headers', 'wb') as f:
        f.write(headers_dict_to_raw(request.headers))
    with self._open(rpath / 'request_body', 'wb') as f:
        f.write(request.body)