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
@pytest.mark.issue(1963)
def test_issue1963(en_tokenizer):
    """Test that doc.merge() resizes doc.tensor"""
    doc = en_tokenizer('a b c d')
    doc.tensor = numpy.ones((len(doc), 128), dtype='f')
    with doc.retokenize() as retokenizer:
        retokenizer.merge(doc[0:2])
    assert len(doc) == 3
    assert doc.tensor.shape == (3, 128)