import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def assertEventDataInList(self, events, **kwargs):
    found = False
    for e in events:
        for key, value in kwargs.items():
            try:
                if e[key] != value:
                    break
            except KeyError:
                break
        else:
            found = True
    self.assertTrue(found, 'event with correct values not in list, expected to find event with key-value pairs. Expected: "%(expected)s" Events: "%(events)s"' % {'expected': ','.join(["'%s=%s'" % (k, v) for k, v in kwargs.items()]), 'events': events})