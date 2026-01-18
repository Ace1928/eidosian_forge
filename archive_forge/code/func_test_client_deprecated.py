import io
import logging
from testtools import matchers
from keystoneclient import exceptions
from keystoneclient import httpclient
from keystoneclient import session
from keystoneclient.tests.unit import utils
def test_client_deprecated(self):
    from keystoneclient import client
    client.HTTPClient