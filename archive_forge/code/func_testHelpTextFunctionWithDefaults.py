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
def testHelpTextFunctionWithDefaults(self):
    component = tc.WithDefaults().triple
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='triple'))
    self.assertIn('NAME\n    triple', help_screen)
    self.assertIn('SYNOPSIS\n    triple <flags>', help_screen)
    self.assertNotIn('DESCRIPTION', help_screen)
    self.assertIn('FLAGS\n    -c, --count=COUNT\n        Default: 0', help_screen)
    self.assertNotIn('NOTES', help_screen)