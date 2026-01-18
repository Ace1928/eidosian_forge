import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_get_members_names_exclude_hidden(self):
    obj = TestObject()
    members = list(reflection.get_member_names(obj, exclude_hidden=True))
    self.assertEqual(['hi'], members)