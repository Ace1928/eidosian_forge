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
@pytest.mark.issue(10086)
def test_issue10086(en_tokenizer):
    """Test special case works when part of infix substring."""
    text = "No--don't see"
    en_tokenizer.faster_heuristics = False
    doc = en_tokenizer(text)
    assert "n't" in [w.text for w in doc]
    assert 'do' in [w.text for w in doc]
    en_tokenizer.faster_heuristics = True
    doc = en_tokenizer(text)
    assert "don't" in [w.text for w in doc]