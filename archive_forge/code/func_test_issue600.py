import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT
from spacy.lookups import Lookups
from spacy.tokens import Doc
from spacy.util import OOV_RANK
from spacy.vocab import Vocab
@pytest.mark.issue(600)
def test_issue600():
    vocab = Vocab(tag_map={'NN': {'pos': 'NOUN'}})
    doc = Doc(vocab, words=['hello'])
    doc[0].tag_ = 'NN'