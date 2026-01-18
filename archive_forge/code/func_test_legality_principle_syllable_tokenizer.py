from typing import List, Tuple
import pytest
from nltk.tokenize import (
def test_legality_principle_syllable_tokenizer(self):
    """
        Test LegalitySyllableTokenizer tokenizer.
        """
    from nltk.corpus import words
    test_word = 'wonderful'
    tokenizer = LegalitySyllableTokenizer(words.words())
    tokens = tokenizer.tokenize(test_word)
    assert tokens == ['won', 'der', 'ful']