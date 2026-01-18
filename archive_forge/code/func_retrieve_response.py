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
def retrieve_response(self, spider: Spider, request: Request):
    """Return response if present in cache, or None otherwise."""
    metadata = self._read_meta(spider, request)
    if metadata is None:
        return
    rpath = Path(self._get_request_path(spider, request))
    with self._open(rpath / 'response_body', 'rb') as f:
        body = f.read()
    with self._open(rpath / 'response_headers', 'rb') as f:
        rawheaders = f.read()
    url = metadata.get('response_url')
    status = metadata['status']
    headers = Headers(headers_raw_to_dict(rawheaders))
    respcls = responsetypes.from_args(headers=headers, url=url, body=body)
    response = respcls(url=url, headers=headers, status=status, body=body)
    return response