from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testMethodCompletions(self):
    completions = completion.Completions(tc.NoDefaults().double)
    self.assertNotIn('--self', completions)
    self.assertIn('--count', completions)