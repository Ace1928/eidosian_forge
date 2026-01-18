import collections
import collections.abc
import datetime
import functools
import io
import ipaddress
import itertools
import json
from unittest import mock
from xmlrpc import client as xmlrpclib
import netaddr
from oslo_i18n import fixture
from oslotest import base as test_base
from oslo_serialization import jsonutils
def test_loads_with_kwargs(self):
    jsontext = '{"foo": 3}'
    result = jsonutils.loads(jsontext, parse_int=lambda x: 5)
    self.assertEqual(5, result['foo'])