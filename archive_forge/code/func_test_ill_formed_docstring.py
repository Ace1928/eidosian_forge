from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_ill_formed_docstring(self):
    docstring = 'Docstring summary.\n\n    args: raises ::\n    :\n    pathological docstrings should not fail, and ideally should behave\n    reasonably.\n    '
    docstrings.parse(docstring)