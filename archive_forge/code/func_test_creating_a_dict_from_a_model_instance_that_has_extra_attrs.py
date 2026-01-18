from sqlalchemy.ext import declarative
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import utils
def test_creating_a_dict_from_a_model_instance_that_has_extra_attrs(self):
    expected = {'id': utils.new_uuid(), 'text': utils.new_uuid()}
    m = TestModel(id=expected['id'], text=expected['text'])
    m.extra = 'this should not be in the dictionary'
    self.assertEqual(expected, m.to_dict())