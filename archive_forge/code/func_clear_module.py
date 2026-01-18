import logging
import sys
import urllib.parse as urlparse
import uuid
import fixtures
from oslo_serialization import jsonutils
import requests
import requests_mock
from requests_mock.contrib import fixture
import testscenarios
import testtools
from keystoneclient.tests.unit import client_fixtures
def clear_module(self):
    cleared_modules = {}
    for fullname in list(sys.modules):
        if fullname == self.module or fullname.startswith(self.module + '.'):
            cleared_modules[fullname] = sys.modules.pop(fullname)
    return cleared_modules