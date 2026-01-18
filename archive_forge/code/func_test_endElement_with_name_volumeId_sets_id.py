from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_endElement_with_name_volumeId_sets_id(self):
    self.volume_attribute.endElement('volumeId', 'some_value', None)
    self.assertEqual(self.volume_attribute.id, 'some_value')