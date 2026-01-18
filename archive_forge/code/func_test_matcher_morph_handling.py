import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_morph_handling(en_vocab):
    matcher = Matcher(en_vocab)
    pattern1 = [{'MORPH': {'IN': ['Feat1=Val1|Feat2=Val2']}}]
    pattern2 = [{'MORPH': {'IN': ['Feat2=Val2|Feat1=Val1']}}]
    matcher.add('M', [pattern1])
    matcher.add('N', [pattern2])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat2=Val2|Feat1=Val1')
    assert len(matcher(doc)) == 2
    doc[0].set_morph('Feat1=Val1|Feat2=Val2')
    assert len(matcher(doc)) == 2
    matcher = Matcher(en_vocab)
    pattern1 = [{'MORPH': {'IS_SUPERSET': ['Feat1=Val1', 'Feat2=Val2']}}]
    pattern2 = [{'MORPH': {'IS_SUPERSET': ['Feat1=Val1', 'Feat1=Val3', 'Feat2=Val2']}}]
    matcher.add('M', [pattern1])
    matcher.add('N', [pattern2])
    doc = Doc(en_vocab, words=['a', 'b', 'c'])
    assert len(matcher(doc)) == 0
    doc[0].set_morph('Feat2=Val2,Val3|Feat1=Val1')
    assert len(matcher(doc)) == 1
    doc[0].set_morph('Feat1=Val1,Val3|Feat2=Val2')
    assert len(matcher(doc)) == 2