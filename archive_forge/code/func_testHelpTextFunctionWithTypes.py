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
@testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
def testHelpTextFunctionWithTypes(self):
    component = tc.py3.WithTypes().double
    help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
    self.assertIn('NAME\n    double', help_screen)
    self.assertIn('SYNOPSIS\n    double COUNT', help_screen)
    self.assertIn('DESCRIPTION', help_screen)
    self.assertIn('POSITIONAL ARGUMENTS\n    COUNT\n        Type: float', help_screen)
    self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)