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
def testHelpScreenForFunctionDocstringWithLineBreak(self):
    component = tc.ClassWithMultilineDocstring.example_generator
    t = trace.FireTrace(component, name='example_generator')
    help_output = helptext.HelpText(component, t)
    expected_output = '\n    NAME\n        example_generator - Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    SYNOPSIS\n        example_generator N\n\n    DESCRIPTION\n        Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    POSITIONAL ARGUMENTS\n        N\n            The upper limit of the range to generate, from 0 to `n` - 1.\n\n    NOTES\n        You can also use flags syntax for POSITIONAL ARGUMENTS'
    self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())