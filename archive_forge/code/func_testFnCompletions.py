from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testFnCompletions(self):

    def example(one, two, three):
        return (one, two, three)
    completions = completion.Completions(example)
    self.assertIn('--one', completions)
    self.assertIn('--two', completions)
    self.assertIn('--three', completions)