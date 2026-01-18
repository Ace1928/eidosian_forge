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
def testHelpScreen(self):
    component = tc.ClassWithDocstring()
    t = trace.FireTrace(component, name='ClassWithDocstring')
    help_output = helptext.HelpText(component, t)
    expected_output = '\nNAME\n    ClassWithDocstring - Test class for testing help text output.\n\nSYNOPSIS\n    ClassWithDocstring COMMAND | VALUE\n\nDESCRIPTION\n    This is some detail description of this test class.\n\nCOMMANDS\n    COMMAND is one of the following:\n\n     print_msg\n       Prints a message.\n\nVALUES\n    VALUE is one of the following:\n\n     message\n       The default message to print.'
    self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())