from unittest import mock
import uuid
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from oslo_config import cfg
from oslotest import base
from glance_store.common import cinder_utils
@mock.patch('time.sleep', new=mock.Mock())
def test_attachment_create_retries(self):
    fake_attach_id = 'fake-attach-id'
    self.fake_client.attachments.create.side_effect = [cinder_exception.BadRequest(400), cinder_exception.BadRequest(400), fake_attach_id]
    fake_attachment_id = self.volume_api.attachment_create(self.fake_client, self.fake_vol_id)
    self.assertEqual(fake_attach_id, fake_attachment_id)
    self.fake_client.attachments.create.assert_has_calls([mock.call(self.fake_vol_id, None, mode=None), mock.call(self.fake_vol_id, None, mode=None), mock.call(self.fake_vol_id, None, mode=None)])