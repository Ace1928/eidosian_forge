import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_extract_groups(self):
    group1 = {'name': 'group1', 'domain': self.domain_id_mock}
    group_by_domain = {self.domain_id_mock: [group1]}
    result = utils.RuleProcessor(self.mapping_id_mock, self.attribute_mapping_schema_1_0).extract_groups(group_by_domain)
    self.assertEqual([group1], list(result))