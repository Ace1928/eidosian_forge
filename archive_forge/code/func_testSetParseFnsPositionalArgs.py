from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFnsPositionalArgs(self):
    self.assertEqual(core.Fire(NoDefaults, command=['quadruple', '5']), 20)