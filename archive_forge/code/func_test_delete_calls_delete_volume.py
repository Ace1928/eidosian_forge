from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_delete_calls_delete_volume(self):
    self.volume_one.connection = mock.Mock()
    self.volume_one.delete()
    self.volume_one.connection.delete_volume.assert_called_with(1, dry_run=False)