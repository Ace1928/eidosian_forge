import pytest
from spacy import util
from spacy.tokens import Doc
from spacy.vocab import Vocab
def test_empty_doc(vocab):
    doc = Doc(vocab)
    assert len(doc) == 0