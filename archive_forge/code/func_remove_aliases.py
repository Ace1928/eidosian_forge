from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def remove_aliases(user_params, metadata):
    if not user_params:
        return user_params
    if isinstance(user_params, str) or isinstance(user_params, int):
        return user_params
    if isinstance(user_params, list):
        new_params = []
        for item in user_params:
            new_params.append(remove_aliases(item, metadata))
        return new_params
    replace_key = {'fmgr_message': 'message', 'fmgr_syslog_facility': 'syslog-facility', 'd80211d': '80211d', 'd80211k': '80211k', 'd80211v': '80211v'}
    new_params = {}
    for param_name, param_data in metadata.items():
        if user_params.get(param_name, None) is None:
            continue
        real_param_name = replace_key.get(param_name, param_name)
        if 'options' in param_data:
            new_params[real_param_name] = remove_aliases(user_params[param_name], param_data['options'])
        else:
            new_params[real_param_name] = user_params[param_name]
    return new_params