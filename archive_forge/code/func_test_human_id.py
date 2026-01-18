from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_human_id(self):
    r = base.Resource(None, {'name': '1'})
    self.assertIsNone(r.human_id)
    r = HumanResource(None, {'name': '1'})
    self.assertEqual('1', r.human_id)
    r = HumanResource(None, {'name': None})
    self.assertIsNone(r.human_id)