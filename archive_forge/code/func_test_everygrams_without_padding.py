import pytest
from nltk.util import everygrams
def test_everygrams_without_padding(everygram_input):
    expected_output = [('a',), ('a', 'b'), ('a', 'b', 'c'), ('b',), ('b', 'c'), ('c',)]
    output = list(everygrams(everygram_input))
    assert output == expected_output