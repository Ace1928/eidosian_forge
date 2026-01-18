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
def test_message_with_named_param(self):
    msg = self.trans_fixture.lazy('A message with params: %(param)s')
    msg = msg % {'param': 'hello'}
    ret = jsonutils.to_primitive(msg)
    self.assertEqual(msg, ret)