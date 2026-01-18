from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
def dummy_app(environ, response):
    res = webob_response.Response()
    return res(environ, response)