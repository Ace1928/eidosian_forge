from __future__ import absolute_import, division, print_function
import re
from ansible.errors import AnsibleFilterError
def keep_keys_from_dict_n_list(data, target, matching_parameter):
    if isinstance(data, list):
        list_data = [keep_keys_from_dict_n_list(each, target, matching_parameter) for each in data]
        return list_data
    if isinstance(data, dict):
        keep = {}
        for k, val in data.items():
            for key in target:
                match = None
                if not isinstance(val, (list, dict)):
                    if matching_parameter == 'regex':
                        match = re.match(key, k)
                        if match:
                            keep[k] = val
                    elif matching_parameter == 'starts_with':
                        if k.startswith(key):
                            keep[k], match = (val, True)
                    elif matching_parameter == 'ends_with':
                        if k.endswith(key):
                            keep[k], match = (val, True)
                    elif k == key:
                        keep[k], match = (val, True)
                else:
                    list_data = keep_keys_from_dict_n_list(val, target, matching_parameter)
                    if isinstance(list_data, list) and (not match):
                        list_data = [r for r in list_data if r not in ([], {})]
                        if all((isinstance(s, str) for s in list_data)):
                            continue
                    if list_data in ([], {}):
                        continue
                    keep[k] = list_data
        return keep
    return data