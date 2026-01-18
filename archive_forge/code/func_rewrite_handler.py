import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import celery
import tornado.web
from tornado import ioloop
from tornado.httpserver import HTTPServer
from tornado.web import url
from .urls import handlers as default_handlers
from .events import Events
from .inspector import Inspector
from .options import default_options
def rewrite_handler(handler, url_prefix):
    if isinstance(handler, url):
        return url('/{}{}'.format(url_prefix.strip('/'), handler.regex.pattern), handler.handler_class, handler.kwargs, handler.name)
    return ('/{}{}'.format(url_prefix.strip('/'), handler[0]), handler[1])