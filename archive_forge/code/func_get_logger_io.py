import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def get_logger_io(name):
    logger_name = 'keystoneauth.session.{name}'.format(name=name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    string_io = io.StringIO()
    handler = logging.StreamHandler(string_io)
    logger.addHandler(handler)
    return string_io