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
def testHelpTextFunctionWithKwargs(self):
    component = tc.fn_with_kwarg
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
    self.assertIn('NAME\n    text', help_screen)
    self.assertIn('SYNOPSIS\n    text ARG1 ARG2 <flags>', help_screen)
    self.assertIn('DESCRIPTION\n    Function with kwarg', help_screen)
    self.assertIn('FLAGS\n    --arg3\n        Description of arg3.\n    Additional undocumented flags may also be accepted.', help_screen)