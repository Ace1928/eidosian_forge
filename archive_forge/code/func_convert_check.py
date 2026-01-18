from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent
from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError
def convert_check(self, before, expected, stages=(1, 2), all_imports=False, ignore_imports=True, from3=False, run=True, conservative=False):
    """
        Convenience method that calls convert() and compare().

        Reformats the code blocks automatically using the reformat_code()
        function.

        If all_imports is passed, we add the appropriate import headers
        for the stage(s) selected to the ``expected`` code-block, so they
        needn't appear repeatedly in the test code.

        If ignore_imports is True, ignores the presence of any lines
        beginning:

            from __future__ import ...
            from future import ...

        for the purpose of the comparison.
        """
    output = self.convert(before, stages=stages, all_imports=all_imports, from3=from3, run=run, conservative=conservative)
    if all_imports:
        headers = self.headers2 if 2 in stages else self.headers1
    else:
        headers = ''
    reformatted = reformat_code(expected)
    if headers in reformatted:
        headers = ''
    self.compare(output, headers + reformatted, ignore_imports=ignore_imports)