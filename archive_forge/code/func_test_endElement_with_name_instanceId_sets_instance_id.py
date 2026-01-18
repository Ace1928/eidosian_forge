from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
def test_endElement_with_name_instanceId_sets_instance_id(self):
    return self.check_that_attribute_has_been_set('instanceId', 1, 'instance_id')