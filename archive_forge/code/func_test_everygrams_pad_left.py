import pytest
from nltk.util import everygrams
def test_everygrams_pad_left(everygram_input):
    expected_output = [(None,), (None, None), (None, None, 'a'), (None,), (None, 'a'), (None, 'a', 'b'), ('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('c',)]
    output = list(everygrams(everygram_input, max_len=3, pad_left=True))
    assert output == expected_output