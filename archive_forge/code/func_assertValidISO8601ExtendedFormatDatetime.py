import datetime
import http.client
import oslo_context.context
from oslo_serialization import jsonutils
from testtools import matchers
import uuid
import webtest
from keystone.common import authorization
from keystone.common import cache
from keystone.common import provider_api
from keystone.common.validation import validators
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import rest
def assertValidISO8601ExtendedFormatDatetime(self, dt):
    try:
        return datetime.datetime.strptime(dt, TIME_FORMAT)
    except Exception:
        msg = '%s is not a valid ISO 8601 extended format date time.' % dt
        raise AssertionError(msg)