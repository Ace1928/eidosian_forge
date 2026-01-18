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
def testUsageOutputFunctionWithDocstring(self):
    component = tc.multiplier_with_docstring
    t = trace.FireTrace(component, name='multiplier_with_docstring')
    usage_output = helptext.UsageText(component, trace=t, verbose=False)
    expected_output = '\n    Usage: multiplier_with_docstring NUM <flags>\n      optional flags:        --rate\n\n    For detailed information on this command, run:\n      multiplier_with_docstring --help'
    self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)