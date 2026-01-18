from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
import re
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
def get_sgt_by_name(self, name):
    if not name:
        return None
    try:
        gen_items_responses = self.ise.exec(family='filter_policy', function='get_filter_policy_generator')
        for items_response in gen_items_responses:
            items = items_response.response['SearchResult']['resources']
            result = get_dict_result(items, 'name', name)
            if result:
                return result
    except (TypeError, AttributeError) as e:
        self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
    except Exception:
        result = None
    return result