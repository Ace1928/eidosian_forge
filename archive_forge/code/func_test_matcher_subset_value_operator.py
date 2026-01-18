import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_subset_value_operator(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'MORPH': {'IS_SUBSET': ['Feat=Val', 'Feat2=Val2']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 3
    doc[0].set_morph('Feat=Val')
    assert len(matcher(doc)) == 3
    doc[0].set_morph('Feat=Val|Feat2=Val2')
    assert len(matcher(doc)) == 3
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3')
    assert len(matcher(doc)) == 2
    doc[0].set_morph('Feat=Val|Feat2=Val2|Feat3=Val3|Feat4=Val4')
    assert len(matcher(doc)) == 2
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUBSET': ['A', 'B']}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 1
    matcher = Matcher(en_vocab)
    pattern = [{'TAG': {'IS_SUBSET': []}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0].tag_ = 'A'
    assert len(matcher(doc)) == 0
    Token.set_extension('ext', default=[])
    matcher = Matcher(en_vocab)
    pattern = [{'_': {'ext': {'IS_SUBSET': ['A', 'B']}}}]
    matcher.add('M', [pattern])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    doc[0]._.ext = ['A']
    doc[1]._.ext = ['C', 'D']
    assert len(matcher(doc)) == 2