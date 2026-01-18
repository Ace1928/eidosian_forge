from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import textwrap
from fire import formatting
from fire import helptext
from fire import test_components as tc
from fire import testutils
from fire import trace
import six
def testHelpTextObjectWithGroupAndValues(self):
    component = tc.TypedProperties()
    t = trace.FireTrace(component, name='TypedProperties')
    help_screen = helptext.HelpText(component=component, trace=t, verbose=True)
    print(help_screen)
    self.assertIn('GROUPS', help_screen)
    self.assertIn('GROUP is one of the following:', help_screen)
    self.assertIn('charlie\n       Class with functions that have default arguments.', help_screen)
    self.assertIn('VALUES', help_screen)
    self.assertIn('VALUE is one of the following:', help_screen)
    self.assertIn('alpha', help_screen)