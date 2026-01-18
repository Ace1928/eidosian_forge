from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def diff_object(self, other):
    diff_dict = {}
    for attribute in self.attribute_values_processed:
        if attribute not in self.readwrite_attrs:
            continue
        if self.attribute_values_processed[attribute] is None:
            continue
        if hasattr(other, attribute):
            attribute_value = getattr(other, attribute)
        else:
            diff_dict[attribute] = 'missing from other'
            continue
        param_type = self.attribute_values_processed[attribute].__class__
        if attribute_value is None or param_type(attribute_value) != self.attribute_values_processed[attribute]:
            str_tuple = (type(self.attribute_values_processed[attribute]), self.attribute_values_processed[attribute], type(attribute_value), attribute_value)
            diff_dict[attribute] = 'difference. ours: (%s) %s other: (%s) %s' % str_tuple
    return diff_dict