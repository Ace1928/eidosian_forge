import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_docstring_decorator(self):
    self.maxDiff = None

    @document_kwargs_from_configdict('CONFIG')
    class ExampleClass(object):
        CONFIG = ExampleConfig()

        @document_kwargs_from_configdict(CONFIG)
        def __init__(self):
            """A simple docstring"""

        @document_kwargs_from_configdict(CONFIG, doc='A simple docstring\n', visibility=USER_OPTION)
        def fcn(self):
            pass
    ref = '\nKeyword Arguments\n-----------------\noption_1: int, default=5\n    The first configuration option\n\nsolver_options: dict, optional\n\n    solver_option_1: float, default=1\n        [DEVELOPER option]\n\n        The first solver configuration option\n\n    solver_option_2: float, default=1\n        The second solver configuration option\n\n        With a very long line containing wrappable text in a long, silly\n        paragraph with little actual information.\n        #) but a bulleted list\n        #) with two bullets\n\n    solver_option_3: float, default=1\n        The third solver configuration option\n\n           This has a leading newline and a very long line containing\n           wrappable text in a long, silly paragraph with little actual\n           information.\n\n        .. and_a_list::\n           #) but a bulleted list\n           #) with two bullets\n\noption_2: int, default=5\n    The second solver configuration option with a very long line\n    containing wrappable text in a long, silly paragraph with little\n    actual information.'
    self.assertEqual(ExampleClass.__doc__, ref.lstrip())
    self.assertEqual(ExampleClass.__init__.__doc__, 'A simple docstring\n' + ref)
    ref = '\nKeyword Arguments\n-----------------\noption_1: int, default=5\n    The first configuration option\n\nsolver_options: dict, optional\n\n    solver_option_2: float, default=1\n        The second solver configuration option\n\n        With a very long line containing wrappable text in a long, silly\n        paragraph with little actual information.\n        #) but a bulleted list\n        #) with two bullets\n\n    solver_option_3: float, default=1\n        The third solver configuration option\n\n           This has a leading newline and a very long line containing\n           wrappable text in a long, silly paragraph with little actual\n           information.\n\n        .. and_a_list::\n           #) but a bulleted list\n           #) with two bullets\n\noption_2: int, default=5\n    The second solver configuration option with a very long line\n    containing wrappable text in a long, silly paragraph with little\n    actual information.'
    self.assertEqual(ExampleClass.fcn.__doc__, 'A simple docstring\n' + ref)
    ref = '\nKeyword Arguments\n-----------------\noption_1: int, default=5\n    The first configuration option\n\nsolver_options: dict, optional\n\n    solver_option_2: float, default=1\n        The second solver configuration option\n\n        With a very long line containing wrappable text in a long, silly paragraph with little actual information.\n        #) but a bulleted list\n        #) with two bullets\n\n    solver_option_3: float, default=1\n        The third solver configuration option\n\n           This has a leading newline and a very long line containing wrappable text in a long, silly paragraph with little actual information.\n\n        .. and_a_list::\n           #) but a bulleted list\n           #) with two bullets\n\noption_2: int, default=5\n    The second solver configuration option with a very long line containing wrappable text in a long, silly paragraph with little actual information.'
    with LoggingIntercept() as LOG:
        self.assertEqual(add_docstring_list('', ExampleClass.CONFIG), ref)
    self.assertIn('add_docstring_list is deprecated', LOG.getvalue())