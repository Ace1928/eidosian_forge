import asyncio
import concurrent.futures
import threading
from wsgiref.validate import validator
from tornado.routing import RuleRouter
from tornado.testing import AsyncHTTPTestCase, gen_test
from tornado.wsgi import WSGIContainer
def barrier_wsgi_app(self, environ, start_response):
    self.respond_plain(start_response)
    try:
        n = self.barrier.wait()
    except threading.BrokenBarrierError:
        return [b'broken barrier']
    else:
        return [b'ok %d' % n]