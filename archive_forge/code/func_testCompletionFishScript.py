from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testCompletionFishScript(self):
    commands = [['run'], ['halt'], ['halt', '--now']]
    script = completion._FishScript(name='command', commands=commands)
    self.assertIn('command', script)
    self.assertIn('halt', script)
    self.assertIn('-l now', script)