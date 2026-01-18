import json
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
from requests_mock.contrib import fixture as rm_fixture
from urllib import parse as urlparse
from oslo_policy import _external
from oslo_policy import opts
from oslo_policy.tests import base
def decode_post_data(self, post_data):
    result = {}
    for item in post_data.split('&'):
        key, _sep, value = item.partition('=')
        result[key] = jsonutils.loads(urlparse.unquote_plus(value))
    return result