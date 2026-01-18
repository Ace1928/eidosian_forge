from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFn(self):
    self.assertEqual(core.Fire(WithVarArgs, command=['example7', '1', '--arg2=2', '3', '4', '--kwarg=5']), ('1', '2', ('3', '4'), {'kwarg': '5'}))