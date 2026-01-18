from __future__ import (absolute_import, division, print_function)
from ansible.module_utils import basic
def filter_out_none_entries(self, list_or_dict):
    """take a dict or list as input and return a dict/list without keys/elements whose values are None
           skip empty dicts or lists.
        """
    if isinstance(list_or_dict, dict):
        result = {}
        for key, value in list_or_dict.items():
            if isinstance(value, (list, dict)):
                sub = self.filter_out_none_entries(value)
                if sub:
                    result[key] = sub
            elif value is not None:
                result[key] = value
        return result
    if isinstance(list_or_dict, list):
        alist = []
        for item in list_or_dict:
            if isinstance(item, (list, dict)):
                sub = self.filter_out_none_entries(item)
                if sub:
                    alist.append(sub)
            elif item is not None:
                alist.append(item)
        return alist
    raise TypeError('unexpected type %s' % type(list_or_dict))