import hashlib
import json
import warnings
from typing import (
from urllib.parse import urlunparse
from weakref import WeakKeyDictionary
from w3lib.http import basic_auth_header
from w3lib.url import canonicalize_url
from scrapy import Request, Spider
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_bytes, to_unicode
def request_from_dict(d: dict, *, spider: Optional[Spider]=None) -> Request:
    """Create a :class:`~scrapy.Request` object from a dict.

    If a spider is given, it will try to resolve the callbacks looking at the
    spider for methods with the same name.
    """
    request_cls: Type[Request] = load_object(d['_class']) if '_class' in d else Request
    kwargs = {key: value for key, value in d.items() if key in request_cls.attributes}
    if d.get('callback') and spider:
        kwargs['callback'] = _get_method(spider, d['callback'])
    if d.get('errback') and spider:
        kwargs['errback'] = _get_method(spider, d['errback'])
    return request_cls(**kwargs)