import warnings
import weakref
import numpy
import pytest
from numpy.testing import assert_array_equal
from thinc.api import NumpyOps, get_current_ops
from spacy.attrs import (
from spacy.lang.en import English
from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from spacy.lexeme import Lexeme
from spacy.tokens import Doc, Span, SpanGroup, Token
from spacy.vocab import Vocab
from .test_underscore import clean_underscore  # noqa: F401
@pytest.mark.issue(3962)
def test_issue3962_long(en_vocab):
    """Ensure that as_doc does not result in out-of-bound access of tokens.
    This is achieved by setting the head to itself if it would lie out of the span otherwise."""
    words = ['He', 'jests', 'at', 'scars', '.', 'They', 'never', 'felt', 'a', 'wound', '.']
    heads = [1, 1, 1, 2, 1, 7, 7, 7, 9, 7, 7]
    deps = ['nsubj', 'ROOT', 'prep', 'pobj', 'punct', 'nsubj', 'neg', 'ROOT', 'det', 'dobj', 'punct']
    two_sent_doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    span2 = two_sent_doc[1:7]
    doc2 = span2.as_doc()
    doc2_json = doc2.to_json()
    assert doc2_json
    assert doc2[0].head.text == 'jests'
    assert doc2[0].dep_ == 'ROOT'
    assert doc2[1].head.text == 'jests'
    assert doc2[1].dep_ == 'prep'
    assert doc2[2].head.text == 'at'
    assert doc2[2].dep_ == 'pobj'
    assert doc2[3].head.text == 'jests'
    assert doc2[3].dep_ == 'punct'
    assert doc2[4].head.text == 'They'
    assert doc2[4].dep_ == 'dep'
    assert doc2[4].head.text == 'They'
    assert doc2[4].dep_ == 'dep'
    sents = list(doc2.sents)
    assert len(sents) == 2
    assert sents[0].text == 'jests at scars .'
    assert sents[1].text == 'They never'