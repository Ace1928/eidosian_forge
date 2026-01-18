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
@pytest.mark.issue(3002)
def test_issue3002():
    """Test that the tokenizer doesn't hang on a long list of dots"""
    nlp = German()
    doc = nlp('880.794.982.218.444.893.023.439.794.626.120.190.780.624.990.275.671 ist eine lange Zahl')
    assert len(doc) == 5