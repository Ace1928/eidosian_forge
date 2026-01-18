from __future__ import absolute_import, division, print_function
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_monitor import DOCUMENTATION
def configure_module_api(self, conn_request, config):
    before = []
    after = []
    changed = False
    defaults = {'disabled': False, 'host': '$decideOnStartup', 'index': 'default'}
    remove_from_diff_compare = ['check_path', 'check_index', 'ignore_older_than', 'time_before_close', 'rename_source']
    for want_conf in config:
        have_conf = self.search_for_resource_name(conn_request, want_conf['name'])
        if have_conf:
            want_conf = set_defaults(want_conf, defaults)
            want_conf = utils.remove_empties(want_conf)
            diff = utils.dict_diff(have_conf, want_conf)
            if self._task.args['state'] == 'replaced':
                diff2 = utils.dict_diff(want_conf, have_conf)
                if len(diff) or len(diff2):
                    diff.update(diff2)
            if diff:
                diff = remove_get_keys_from_payload_dict(diff, remove_from_diff_compare)
                if diff:
                    before.append(have_conf)
                    if self._task.args['state'] == 'merged':
                        want_conf = utils.remove_empties(utils.dict_merge(have_conf, want_conf))
                        want_conf = remove_get_keys_from_payload_dict(want_conf, remove_from_diff_compare)
                        changed = True
                        payload = map_obj_to_params(want_conf, self.key_transform)
                        url = '{0}/{1}'.format(self.api_object, quote_plus(payload.pop('name')))
                        api_response = conn_request.create_update(url, data=payload)
                        response_json = self.map_params_to_object(api_response['entry'][0])
                        after.append(response_json)
                    elif self._task.args['state'] == 'replaced':
                        conn_request.delete_by_path('{0}/{1}'.format(self.api_object, quote_plus(want_conf['name'])))
                        changed = True
                        payload = map_obj_to_params(want_conf, self.key_transform)
                        url = '{0}'.format(self.api_object)
                        api_response = conn_request.create_update(url, data=payload)
                        response_json = self.map_params_to_object(api_response['entry'][0])
                        after.append(response_json)
                else:
                    before.append(have_conf)
                    after.append(have_conf)
            else:
                before.append(have_conf)
                after.append(have_conf)
        else:
            changed = True
            want_conf = utils.remove_empties(want_conf)
            payload = map_obj_to_params(want_conf, self.key_transform)
            url = '{0}'.format(self.api_object)
            api_response = conn_request.create_update(url, data=payload)
            response_json = self.map_params_to_object(api_response['entry'][0])
            after.extend(before)
            after.append(response_json)
    if not changed:
        after = None
    res_config = {}
    res_config['after'] = after
    res_config['before'] = before
    return (res_config, changed)