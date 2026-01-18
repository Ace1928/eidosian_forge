from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str
def scraped(self, item: Any, response: Union[Response, Failure], spider: Spider) -> dict:
    """Logs a message when an item is scraped by a spider."""
    src: Any
    if isinstance(response, Failure):
        src = response.getErrorMessage()
    else:
        src = response
    return {'level': logging.DEBUG, 'msg': SCRAPEDMSG, 'args': {'src': src, 'item': item}}