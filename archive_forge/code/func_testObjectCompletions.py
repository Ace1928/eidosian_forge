from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testObjectCompletions(self):
    completions = completion.Completions(tc.NoDefaults())
    self.assertIn('double', completions)
    self.assertIn('triple', completions)