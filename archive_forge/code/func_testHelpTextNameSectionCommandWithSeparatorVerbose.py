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
def testHelpTextNameSectionCommandWithSeparatorVerbose(self):
    component = tc.WithDefaults().double
    t = trace.FireTrace(component, name='double', separator='-')
    t.AddSeparator()
    help_screen = helptext.HelpText(component=component, trace=t, verbose=True)
    self.assertIn('double -', help_screen)
    self.assertIn('double - -', help_screen)