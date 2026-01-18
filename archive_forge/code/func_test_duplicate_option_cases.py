import uuid
from keystone.common import resource_options
from keystone.tests import unit
def test_duplicate_option_cases(self):
    option_id_str_valid = 'test'
    registry = resource_options.ResourceOptionRegistry(option_id_str_valid)
    option_name_unique = uuid.uuid4().hex
    option = resource_options.ResourceOption(option_id_str_valid, option_name_unique)
    option_dup_id = resource_options.ResourceOption(option_id_str_valid, uuid.uuid4().hex)
    option_dup_name = resource_options.ResourceOption(uuid.uuid4().hex[:4], option_name_unique)
    registry.register_option(option)
    self.assertRaises(ValueError, registry.register_option, option_dup_id)
    self.assertRaises(ValueError, registry.register_option, option_dup_name)
    self.assertIs(1, len(registry.options))
    registry.register_option(option)
    self.assertIs(1, len(registry.options))