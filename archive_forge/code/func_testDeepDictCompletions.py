from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testDeepDictCompletions(self):
    deepdict = {'level1': {'level2': {'level3': {'level4': {}}}}}
    completions = completion.Completions(deepdict)
    self.assertIn('level1', completions)
    self.assertNotIn('level2', completions)