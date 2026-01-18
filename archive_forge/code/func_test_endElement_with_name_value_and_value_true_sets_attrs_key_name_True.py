from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_endElement_with_name_value_and_value_true_sets_attrs_key_name_True(self):
    self.volume_attribute.endElement('value', 'true', None)
    self.assertEqual(self.volume_attribute.attrs['key_name'], True)