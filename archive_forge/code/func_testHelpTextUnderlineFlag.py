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
def testHelpTextUnderlineFlag(self):
    component = tc.WithDefaults().triple
    t = trace.FireTrace(component, name='triple')
    help_screen = helptext.HelpText(component, t)
    self.assertIn(formatting.Bold('NAME') + '\n    triple', help_screen)
    self.assertIn(formatting.Bold('SYNOPSIS') + '\n    triple <flags>', help_screen)
    self.assertIn(formatting.Bold('FLAGS') + '\n    -c, --' + formatting.Underline('count'), help_screen)