import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
@pytest.mark.issue(1061)
def test_issue1061():
    """Test special-case works after tokenizing. Was caching problem."""
    text = 'I like _MATH_ even _MATH_ when _MATH_, except when _MATH_ is _MATH_! but not _MATH_.'
    tokenizer = English().tokenizer
    doc = tokenizer(text)
    assert 'MATH' in [w.text for w in doc]
    assert '_MATH_' not in [w.text for w in doc]
    tokenizer.add_special_case('_MATH_', [{ORTH: '_MATH_'}])
    doc = tokenizer(text)
    assert '_MATH_' in [w.text for w in doc]
    assert 'MATH' not in [w.text for w in doc]
    tokenizer = English().tokenizer
    tokenizer.add_special_case('_MATH_', [{ORTH: '_MATH_'}])
    doc = tokenizer(text)
    assert '_MATH_' in [w.text for w in doc]
    assert 'MATH' not in [w.text for w in doc]