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
def testUsageOutputConstructorWithParameter(self):
    component = tc.InstanceVars
    t = trace.FireTrace(component, name='InstanceVars')
    usage_output = helptext.UsageText(component, trace=t, verbose=False)
    expected_output = '\n    Usage: InstanceVars --arg1=ARG1 --arg2=ARG2\n\n    For detailed information on this command, run:\n      InstanceVars --help'
    self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)