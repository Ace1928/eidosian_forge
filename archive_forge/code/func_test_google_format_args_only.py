from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_google_format_args_only(self):
    docstring = 'One line description.\n\n    Args:\n      arg1: arg1_description\n      arg2: arg2_description\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='One line description.', args=[ArgInfo(name='arg1', description='arg1_description'), ArgInfo(name='arg2', description='arg2_description')])
    self.assertEqual(expected_docstring_info, docstring_info)