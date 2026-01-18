import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def check_ns(self, lines, ns):
    """Validate that the given input lines produce the resulting namespace.

        Note: the input lines are given exactly as they would be typed in an
        auto-indenting environment, as mini_interactive_loop above already does
        auto-indenting and prepends spaces to the input.
        """
    src = mini_interactive_loop(pseudo_input(lines))
    test_ns = {}
    exec(src, test_ns)
    for k, v in ns.items():
        self.assertEqual(test_ns[k], v)