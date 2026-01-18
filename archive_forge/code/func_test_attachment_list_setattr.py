from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
@mock.patch('cinderclient.shell_utils.print_list')
@mock.patch.object(cinderclient.v3.attachments.VolumeAttachmentManager, 'list')
def test_attachment_list_setattr(self, mock_list, mock_print):
    command = '--os-volume-api-version 3.27 attachment-list '
    fake_attachment = [attachments.VolumeAttachment(mock.ANY, attachment) for attachment in fakes.fake_attachment_list['attachments']]
    mock_list.return_value = fake_attachment
    self.run_command(command)
    for attach in fake_attachment:
        setattr(attach, 'server_id', getattr(attach, 'instance'))
    columns = ['ID', 'Volume ID', 'Status', 'Server ID']
    mock_print.assert_called_once_with(fake_attachment, columns, sortby_index=0)