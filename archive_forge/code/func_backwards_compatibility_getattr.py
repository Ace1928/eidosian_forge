import warnings
from logging import Logger, getLogger
from typing import Optional, Type, Union
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.http.request import Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
from scrapy.utils.misc import load_object
from scrapy.utils.python import global_object_name
from scrapy.utils.response import response_status_message
def backwards_compatibility_getattr(self, name):
    if name == 'EXCEPTIONS_TO_RETRY':
        warnings.warn('Attribute RetryMiddleware.EXCEPTIONS_TO_RETRY is deprecated. Use the RETRY_EXCEPTIONS setting instead.', ScrapyDeprecationWarning, stacklevel=2)
        return tuple((load_object(x) if isinstance(x, str) else x for x in Settings().getlist('RETRY_EXCEPTIONS')))
    raise AttributeError(f'{self.__class__.__name__!r} object has no attribute {name!r}')