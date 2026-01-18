from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import sys
import unittest
from six.moves import zip
def requires_features(*features):
    return unittest.skipIf(any((not supports_feature(feature) for feature in features)), 'Tests features which are not supported by this version of python. Missing: %r' % [f for f in features if not supports_feature(f)])