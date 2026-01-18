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
def test_fallback_nasty(self):
    obj = int
    ret = jsonutils.to_primitive(obj)
    self.assertEqual(str(obj), ret)

    def formatter(typeobj):
        return 'type:%s' % typeobj.__name__
    ret = jsonutils.to_primitive(obj, fallback=formatter)
    self.assertEqual('type:int', ret)