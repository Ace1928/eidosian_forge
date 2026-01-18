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
def test_nasties(self):

    def foo():
        pass
    x = [datetime, foo, dir]
    ret = jsonutils.to_primitive(x)
    self.assertEqual(3, len(ret))
    self.assertTrue(ret[0].startswith(u"<module 'datetime' from ") or ret[0].startswith(u"<module 'datetime' (built-in)"))
    self.assertTrue(ret[1].startswith('<function ToPrimitiveTestCase.test_nasties.<locals>.foo at 0x'))
    self.assertEqual('<built-in function dir>', ret[2])