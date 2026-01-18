from __future__ import (absolute_import, division, print_function)
import sys
def has_feature(self, feature_name):
    feature = self.get_feature(feature_name)
    if isinstance(feature, bool):
        return feature
    self.module.fail_json(msg='Error: expected bool type for feature flag: %s' % feature_name)