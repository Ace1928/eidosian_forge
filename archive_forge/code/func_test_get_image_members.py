import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import image_members
def test_get_image_members(self):
    image_member = self.controller.get(IMAGE, MEMBER)
    self.assertEqual(IMAGE, image_member.image_id)
    self.assertEqual(MEMBER, image_member.member_id)