from sqlalchemy.ext import declarative
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import utils
def test_creating_a_dict_from_a_model_instance(self):
    m = TestModel(id=utils.new_uuid(), text=utils.new_uuid())
    d = m.to_dict()
    self.assertEqual(d['id'], m.id)
    self.assertEqual(d['text'], m.text)