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
def testUsageOutput(self):
    component = tc.NoDefaults()
    t = trace.FireTrace(component, name='NoDefaults')
    usage_output = helptext.UsageText(component, trace=t, verbose=False)
    expected_output = '\n    Usage: NoDefaults <command>\n      available commands:    double | triple\n\n    For detailed information on this command, run:\n      NoDefaults --help'
    self.assertEqual(usage_output, textwrap.dedent(expected_output).lstrip('\n'))