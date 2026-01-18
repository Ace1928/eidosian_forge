from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testDictCompletions(self):
    colors = {'red': 'green', 'blue': 'yellow', '_rainbow': True}
    completions = completion.Completions(colors)
    self.assertIn('red', completions)
    self.assertIn('blue', completions)
    self.assertNotIn('green', completions)
    self.assertNotIn('yellow', completions)
    self.assertNotIn('_rainbow', completions)
    self.assertNotIn('True', completions)
    self.assertNotIn(True, completions)