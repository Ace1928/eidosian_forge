import unittest
from os import sys, path
def test_extra_checks(self, error_msg, search_in, feature_name, feature_dict):
    if feature_dict.get('disabled') is not None:
        return
    extra_checks = feature_dict.get('extra_checks', '')
    if not extra_checks:
        return
    if isinstance(extra_checks, str):
        extra_checks = extra_checks.split()
    for f in extra_checks:
        impl_dict = search_in.get(f)
        if not impl_dict or 'disable' in impl_dict:
            continue
        raise AssertionError(error_msg + "in option 'extra_checks', extra test case '%s' already exists as a feature name" % f)