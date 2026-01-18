from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def is_object_difference(self, remote_obj, local_obj):
    for key in local_obj:
        local_value = local_obj[key]
        if local_value is None:
            continue
        remote_value = remote_obj.get(key, None)
        if remote_value is None:
            return True
        if isinstance(local_value, list):
            try:
                if isinstance(remote_value, list):
                    if str(sorted(remote_value)) == str(sorted(local_value)):
                        continue
                elif len(local_value) == 1:
                    if str(remote_value) == str(local_value[0]):
                        continue
            except Exception as e:
                return True
            return True
        elif isinstance(local_value, dict):
            if not isinstance(remote_value, dict):
                return True
            elif self.is_object_difference(remote_value, local_value):
                return True
        else:
            value_string = str(local_value)
            if isinstance(remote_value, list):
                if self.is_same_subnet(remote_value, value_string):
                    continue
                elif len(remote_value) != 1 or str(remote_value[0]) != value_string:
                    return True
            elif str(remote_value) != value_string:
                return True
    return False