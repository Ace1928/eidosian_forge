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
@pytest.mark.issue(2754)
def test_issue2754(en_tokenizer):
    """Test that words like 'a' and 'a.m.' don't get exceptional norm values."""
    a = en_tokenizer('a')
    assert a[0].norm_ == 'a'
    am = en_tokenizer('am')
    assert am[0].norm_ == 'am'