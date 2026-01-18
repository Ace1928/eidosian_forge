from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
def search_obj_in_list(name, lst, key='name'):
    if lst:
        for item in lst:
            if item[key] == name:
                return item
    return None