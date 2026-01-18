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
@pytest.mark.issue(743)
def test_issue743():
    doc = Doc(Vocab(), ['hello', 'world'])
    token = doc[0]
    s = set([token])
    items = list(s)
    assert items[0] is token