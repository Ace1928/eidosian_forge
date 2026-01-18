import unittest
from os import sys, path
def conf_features_partial(self):
    conf_all = self.conf_features
    for feature_name, feature in conf_all.items():
        self.test_feature('attribute conf_features', conf_all, feature_name, feature)
    conf_partial = FakeCCompilerOpt.conf_features_partial(self)
    for feature_name, feature in conf_partial.items():
        self.test_feature('conf_features_partial()', conf_partial, feature_name, feature)
    return conf_partial