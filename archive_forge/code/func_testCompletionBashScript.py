from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import completion
from fire import test_components as tc
from fire import testutils
def testCompletionBashScript(self):
    commands = [['run'], ['halt'], ['halt', '--now']]
    script = completion._BashScript(name='command', commands=commands)
    self.assertIn('command', script)
    self.assertIn('halt', script)
    assert_template = '{command})'
    for last_command in ['command', 'halt']:
        self.assertIn(assert_template.format(command=last_command), script)