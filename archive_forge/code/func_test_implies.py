import unittest
from os import sys, path
def test_implies(self, error_msg, search_in, feature_name, feature_dict):
    if feature_dict.get('disabled') is not None:
        return
    implies = feature_dict.get('implies', '')
    if not implies:
        return
    if isinstance(implies, str):
        implies = implies.split()
    if feature_name in implies:
        raise AssertionError(error_msg + 'feature implies itself')
    for impl in implies:
        impl_dict = search_in.get(impl)
        if impl_dict is not None:
            if 'disable' in impl_dict:
                raise AssertionError(error_msg + "implies disabled feature '%s'" % impl)
            continue
        raise AssertionError(error_msg + "implies non-exist feature '%s'" % impl)