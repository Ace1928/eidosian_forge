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
def test_ipnet_using_netaddr(self):
    thing = {'ip_net': netaddr.IPNetwork('1.2.3.0/24')}
    ret = jsonutils.to_primitive(thing)
    self.assertEqual({'ip_net': '1.2.3.0/24'}, ret)