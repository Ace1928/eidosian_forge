from __future__ import unicode_literals
import contextlib
import logging
import unittest
import sys
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.format import formatter
from cmakelang.parse.common import NodeType
def test_multiline_reflow(self):
    self.do_layout_test('      # This multiline-comment should be reflowed\n      # into a single comment\n      # on one line\n      ', [(NodeType.BODY, 0, 0, 77, [(NodeType.COMMENT, 0, 0, 77, [])])])