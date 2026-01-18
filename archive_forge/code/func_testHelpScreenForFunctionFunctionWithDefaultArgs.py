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
def testHelpScreenForFunctionFunctionWithDefaultArgs(self):
    component = tc.WithDefaults().double
    t = trace.FireTrace(component, name='double')
    help_output = helptext.HelpText(component, t)
    expected_output = '\n    NAME\n        double - Returns the input multiplied by 2.\n\n    SYNOPSIS\n        double <flags>\n\n    DESCRIPTION\n        Returns the input multiplied by 2.\n\n    FLAGS\n        -c, --count=COUNT\n            Default: 0\n            Input number that you want to double.'
    self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())