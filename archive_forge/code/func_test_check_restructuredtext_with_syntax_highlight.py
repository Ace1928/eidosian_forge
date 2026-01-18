import os
import textwrap
import unittest
from distutils.command.check import check, HAS_DOCUTILS
from distutils.tests import support
from distutils.errors import DistutilsSetupError
@unittest.skipUnless(HAS_DOCUTILS, "won't test without docutils")
def test_check_restructuredtext_with_syntax_highlight(self):
    example_rst_docs = []
    example_rst_docs.append(textwrap.dedent("            Here's some code:\n\n            .. code:: python\n\n                def foo():\n                    pass\n            "))
    example_rst_docs.append(textwrap.dedent("            Here's some code:\n\n            .. code-block:: python\n\n                def foo():\n                    pass\n            "))
    for rest_with_code in example_rst_docs:
        pkg_info, dist = self.create_dist(long_description=rest_with_code)
        cmd = check(dist)
        cmd.check_restructuredtext()
        msgs = cmd._check_rst_data(rest_with_code)
        if pygments is not None:
            self.assertEqual(len(msgs), 0)
        else:
            self.assertEqual(len(msgs), 1)
            self.assertEqual(str(msgs[0][1]), 'Cannot analyze code. Pygments package not found.')