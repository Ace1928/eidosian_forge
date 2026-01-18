from twisted.internet.defer import gatherResults
from twisted.trial.unittest import TestCase
from twisted.web.http import NOT_FOUND
from twisted.web.resource import NoResource
from twisted.web.server import Site
from twisted.web.static import Data
from twisted.web.test._util import _render
from twisted.web.test.test_web import DummyRequest
from twisted.web.vhost import NameVirtualHost, VHostMonsterResource, _HostResource
def cbRendered(ignored):
    self.assertEqual(request.responseCode, NOT_FOUND)