from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
def test_enabled(self):
    web.disable()
    web.enable()
    self.assertTrue(web._ENABLED)