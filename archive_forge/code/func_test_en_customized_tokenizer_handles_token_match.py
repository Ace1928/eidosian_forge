import re
import pytest
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
def test_en_customized_tokenizer_handles_token_match(custom_en_tokenizer):
    sentence = 'The 8 and 10-county definitions a-b not used for the greater Southern California Megaregion.'
    context = [word.text for word in custom_en_tokenizer(sentence)]
    assert context == ['The', '8', 'and', '10', '-', 'county', 'definitions', 'a-b', 'not', 'used', 'for', 'the', 'greater', 'Southern', 'California', 'Megaregion', '.']