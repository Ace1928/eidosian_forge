import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import image_members
def test_list_image_members(self):
    image_id = IMAGE
    image_members = self.controller.list(image_id)
    self.assertEqual(IMAGE, image_members[0].image_id)
    self.assertEqual(MEMBER, image_members[0].member_id)