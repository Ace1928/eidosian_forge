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
def testHelpTextInt(self):
    component = 7
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, '7'))
    self.assertIn('NAME\n    7', help_screen)
    self.assertIn('SYNOPSIS\n    7 COMMAND | VALUE', help_screen)
    self.assertNotIn('DESCRIPTION', help_screen)
    self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)
    self.assertIn('VALUES\n    VALUE is one of the following:\n', help_screen)