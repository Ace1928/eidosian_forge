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
def strip_future_imports(self, code):
    """
        Strips any of these import lines:

            from __future__ import <anything>
            from future <anything>
            from future.<anything>
            from builtins <anything>

        or any line containing:
            install_hooks()
        or:
            install_aliases()

        Limitation: doesn't handle imports split across multiple lines like
        this:

            from __future__ import (absolute_import, division, print_function,
                                    unicode_literals)
        """
    output = []
    for line in code.split('\n'):
        if not (line.startswith('from __future__ import ') or line.startswith('from future ') or line.startswith('from builtins ') or ('install_hooks()' in line) or ('install_aliases()' in line) or line.startswith('from future.')):
            output.append(line)
    return '\n'.join(output)