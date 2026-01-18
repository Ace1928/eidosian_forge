from sqlalchemy.ext import declarative
from keystone.common import sql
from keystone.tests import unit
from keystone.tests.unit import utils
def test_creating_a_model_instance_from_an_invalid_dict(self):
    d = {'id': utils.new_uuid(), 'text': utils.new_uuid(), 'extra': None}
    self.assertRaises(TypeError, TestModel.from_dict, d)