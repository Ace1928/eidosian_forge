from typing import Any, Callable, Generator, List, Union, cast
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Spider
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Request, Response
from scrapy.middleware import MiddlewareManager
from scrapy.settings import BaseSettings
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import deferred_from_coro, mustbe_deferred
@inlineCallbacks
def process_exception(failure: Failure) -> Generator[Deferred, Any, Union[Failure, Response, Request]]:
    exception = failure.value
    for method in self.methods['process_exception']:
        method = cast(Callable, method)
        response = (yield deferred_from_coro(method(request=request, exception=exception, spider=spider)))
        if response is not None and (not isinstance(response, (Response, Request))):
            raise _InvalidOutput(f'Middleware {method.__qualname__} must return None, Response or Request, got {type(response)}')
        if response:
            return response
    return failure