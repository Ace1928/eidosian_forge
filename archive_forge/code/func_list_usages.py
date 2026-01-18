from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
def list_usages(self):
    result = []
    try:
        response = self.log_analytics_client.usages.list(self.resource_group, self.name)
        while True:
            result.append(response.next().as_dict())
    except StopIteration:
        pass
    except Exception as exc:
        self.fail('Error when listing usages {0}'.format(exc.message or str(exc)))
    return result