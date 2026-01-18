import http.client as http
from unittest.mock import patch
from oslo_log.fixture import logging_error as log_fixture
from oslo_policy import policy
from oslo_utils.fixture import uuidsentinel as uuids
import testtools
import webob
import glance.api.middleware.cache
import glance.api.policy
from glance.common import exception
from glance import context
from glance.tests.unit import base
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import test_policy
from glance.tests.unit import utils as unit_test_utils
def fake_get_v2_image_metadata(*args, **kwargs):
    image = ImageStub(image_id, request.context.project_id, extra_properties=extra_properties)
    request.environ['api.cache.image'] = image
    return (image, glance.api.policy.ImageTarget(image))