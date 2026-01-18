import argparse
from io import StringIO
import itertools
import logging
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_serialization import jsonutils
import requests
from testtools import matchers
from keystoneclient import adapter
from keystoneclient.auth import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.tests.unit import utils
def test_cert(self):
    tup = (self.CERT, self.KEY)
    self.assertEqual(self._s(cert=tup).cert, tup)
    self.assertEqual(self._s(cert=self.CERT, key=self.KEY).cert, tup)
    self.assertIsNone(self._s(key=self.KEY).cert)