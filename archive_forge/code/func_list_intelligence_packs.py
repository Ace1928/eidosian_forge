from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def list_intelligence_packs(self):
    try:
        response = self.log_analytics_client.intelligence_packs.list(self.resource_group, self.name)
        return [x.as_dict() for x in response]
    except Exception as exc:
        self.fail('Error when listing intelligence packs {0}'.format(exc.message or str(exc)))