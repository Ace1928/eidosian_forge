from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
def change_intelligence(self, key, value):
    try:
        if value:
            self.log_analytics_client.intelligence_packs.enable(self.resource_group, self.name, key)
        else:
            self.log_analytics_client.intelligence_packs.disable(self.resource_group, self.name, key)
    except Exception as exc:
        self.fail('Error when changing intelligence pack {0} - {1}'.format(key, exc.message or str(exc)))