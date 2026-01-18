from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testGeneratorCompletions(self):

    def generator():
        x = 0
        while True:
            yield x
            x += 1
    completions = completion.Completions(generator())
    self.assertEqual(completions, [])