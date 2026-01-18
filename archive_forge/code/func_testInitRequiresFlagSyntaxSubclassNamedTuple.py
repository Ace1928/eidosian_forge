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
def testInitRequiresFlagSyntaxSubclassNamedTuple(self):
    component = tc.SubPoint
    t = trace.FireTrace(component, name='SubPoint')
    usage_output = helptext.UsageText(component, trace=t, verbose=False)
    expected_output = 'Usage: SubPoint --x=X --y=Y'
    self.assertIn(expected_output, usage_output)