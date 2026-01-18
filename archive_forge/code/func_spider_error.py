from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.http import Response
from scrapy.utils.request import referer_str
def spider_error(self, failure: Failure, request: Request, response: Union[Response, Failure], spider: Spider) -> dict:
    """Logs an error message from a spider.

        .. versionadded:: 2.0
        """
    return {'level': logging.ERROR, 'msg': SPIDERERRORMSG, 'args': {'request': request, 'referer': referer_str(request)}}