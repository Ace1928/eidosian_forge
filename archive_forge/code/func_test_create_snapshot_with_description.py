from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_create_snapshot_with_description(self):
    self.volume_one.connection = mock.Mock()
    self.volume_one.create_snapshot('some description')
    self.volume_one.connection.create_snapshot.assert_called_with(1, 'some description', dry_run=False)