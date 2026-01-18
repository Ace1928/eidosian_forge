from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
@settings(max_examples=1000, deadline=1000)
@given(st.text(min_size=1))
@example('This is a one-line docstring.')
def test_fuzz_parse(self, value):
    docstrings.parse(value)