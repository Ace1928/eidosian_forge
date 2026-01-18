from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_attachment_state_no_attach_data_returns_None(self):
    retval = self.volume_two.attachment_state()
    self.assertEqual(retval, None)