import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch
import numpy as np
import pandas
import pytest
import modin.pandas as pd
import modin.utils
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs
@pytest.mark.parametrize('source_doc,to_append,expected', [('One-line doc.', 'One-line message.', 'One-line doc.One-line message.'), ('\n            Regular doc-string\n                With the setted indent style.\n            ', '\n                    Doc-string having different indents\n                        in comparison with the regular one.\n            ', '\n            Regular doc-string\n                With the setted indent style.\n\n            Doc-string having different indents\n                in comparison with the regular one.\n            ')])
def test_append_to_docstring(source_doc, to_append, expected):

    def source_fn():
        pass
    source_fn.__doc__ = source_doc
    result_fn = modin.utils.append_to_docstring(to_append)(source_fn)
    answer = dedent(result_fn.__doc__)
    expected = dedent(expected)
    assert answer == expected