import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_multiline_passthrough(self):
    isp = self.isp

    class CommentTransformer(InputTransformer):

        def __init__(self):
            self._lines = []

        def push(self, line):
            self._lines.append(line + '#')

        def reset(self):
            text = '\n'.join(self._lines)
            self._lines = []
            return text
    isp.physical_line_transforms.insert(0, CommentTransformer())
    for raw, expected in [('a=5', 'a=5#'), ('%ls foo', 'get_ipython().run_line_magic(%r, %r)' % (u'ls', u'foo#')), ('!ls foo\n%ls bar', 'get_ipython().system(%r)\nget_ipython().run_line_magic(%r, %r)' % (u'ls foo#', u'ls', u'bar#')), ('1\n2\n3\n%ls foo\n4\n5', '1#\n2#\n3#\nget_ipython().run_line_magic(%r, %r)\n4#\n5#' % (u'ls', u'foo#'))]:
        out = isp.transform_cell(raw)
        self.assertEqual(out.rstrip(), expected.rstrip())