import unittest
from os import sys, path
def test_group(self, error_msg, search_in, feature_name, feature_dict):
    if feature_dict.get('disabled') is not None:
        return
    group = feature_dict.get('group', '')
    if not group:
        return
    if isinstance(group, str):
        group = group.split()
    for f in group:
        impl_dict = search_in.get(f)
        if not impl_dict or 'disable' in impl_dict:
            continue
        raise AssertionError(error_msg + "in option 'group', '%s' already exists as a feature name" % f)