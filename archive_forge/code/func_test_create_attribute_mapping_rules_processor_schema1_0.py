import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_create_attribute_mapping_rules_processor_schema1_0(self):
    result = utils.create_attribute_mapping_rules_processor(self.attribute_mapping_schema_1_0)
    self.assertIsInstance(result, utils.RuleProcessor)