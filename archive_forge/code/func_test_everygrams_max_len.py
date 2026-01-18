import pytest
from nltk.util import everygrams
def test_everygrams_max_len(everygram_input):
    expected_output = [('a',), ('a', 'b'), ('b',), ('b', 'c'), ('c',)]
    output = list(everygrams(everygram_input, max_len=2))
    assert output == expected_output