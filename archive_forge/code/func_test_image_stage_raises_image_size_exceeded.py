import http.client as http
import io
from unittest import mock
import uuid
from cursive import exception as cursive_exception
import glance_store
from glance_store._drivers import filesystem
from oslo_config import cfg
import webob
import glance.api.policy
import glance.api.v2.image_data
from glance.common import exception
from glance.common import wsgi
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
@mock.patch.object(filesystem.Store, 'add')
def test_image_stage_raises_image_size_exceeded(self, mock_store_add):
    mock_store_add.side_effect = exception.ImageSizeLimitExceeded()
    image_id = str(uuid.uuid4())
    request = unit_test_utils.get_fake_request(roles=['admin', 'member'])
    image = FakeImage(image_id=image_id)
    self.image_repo.result = image
    with mock.patch.object(self.controller, '_unstage'):
        self.assertRaises(webob.exc.HTTPRequestEntityTooLarge, self.controller.stage, request, image_id, 'YYYYYYY', 7)