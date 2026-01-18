import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
def test_image_set_data_upload_and_activate_notification_disabled(self):
    insurance = {'called': False}
    image = ImageStub(image_id=UUID1, name='image-1', status='queued', created_at=DATETIME, updated_at=DATETIME, owner=TENANT1, visibility='public')
    context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
    fake_notifier = unit_test_utils.FakeNotifier()
    image_proxy = glance.notifier.ImageProxy(image, context, fake_notifier)

    def data_iterator():
        fake_notifier.log = []
        yield 'abcde'
        yield 'fghij'
        insurance['called'] = True
    self.config(disabled_notifications=['image.activate', 'image.upload'])
    image_proxy.set_data(data_iterator(), 10)
    self.assertTrue(insurance['called'])
    output_logs = fake_notifier.get_logs()
    self.assertEqual(0, len(output_logs))