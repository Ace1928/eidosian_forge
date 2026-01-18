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
def testHelpTextNameSectionCommandWithSeparator(self):
    component = 9
    t = trace.FireTrace(component, name='int', separator='-')
    t.AddSeparator()
    help_screen = helptext.HelpText(component=component, trace=t, verbose=False)
    self.assertIn('int -', help_screen)
    self.assertNotIn('int - -', help_screen)