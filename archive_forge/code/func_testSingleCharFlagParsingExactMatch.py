from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import fire
from fire import test_components as tc
from fire import testutils
import mock
import six
def testSingleCharFlagParsingExactMatch(self):
    self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-a']), (True, None))
    self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-a=10']), (10, None))
    self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '--a']), (True, None))
    self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-alpha']), (None, True))
    self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-a', '-alpha']), (True, True))