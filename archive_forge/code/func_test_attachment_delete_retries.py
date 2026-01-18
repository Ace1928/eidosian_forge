from unittest import mock
import uuid
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from oslo_config import cfg
from oslotest import base
from glance_store.common import cinder_utils
def test_attachment_delete_retries(self):
    self.fake_client.attachments.delete.side_effect = [apiclient_exception.InternalServerError(), apiclient_exception.InternalServerError(), lambda aid: 'foo']
    self.assertIsNone(self.volume_api.attachment_delete(self.fake_client, self.fake_attach_id))
    self.fake_client.attachments.delete.assert_has_calls([mock.call(self.fake_attach_id), mock.call(self.fake_attach_id), mock.call(self.fake_attach_id)])