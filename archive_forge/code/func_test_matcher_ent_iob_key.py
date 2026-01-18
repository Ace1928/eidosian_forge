import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_matcher_ent_iob_key(en_vocab):
    """Test that patterns with ent_iob works correctly."""
    matcher = Matcher(en_vocab)
    matcher.add('Rule', [[{'ENT_IOB': 'I'}]])
    doc1 = Doc(en_vocab, words=['I', 'visited', 'New', 'York', 'and', 'California'])
    doc1.ents = [Span(doc1, 2, 4, label='GPE'), Span(doc1, 5, 6, label='GPE')]
    doc2 = Doc(en_vocab, words=['I', 'visited', 'my', 'friend', 'Alicia'])
    doc2.ents = [Span(doc2, 4, 5, label='PERSON')]
    matches1 = [doc1[start:end].text for _, start, end in matcher(doc1)]
    matches2 = [doc2[start:end].text for _, start, end in matcher(doc2)]
    assert len(matches1) == 1
    assert matches1[0] == 'York'
    assert len(matches2) == 0
    matcher = Matcher(en_vocab)
    matcher.add('Rule', [[{'ENT_IOB': 'I', 'OP': '+'}]])
    doc = Doc(en_vocab, words=['I', 'visited', 'my', 'friend', 'Anna', 'Maria', 'Esperanza'])
    doc.ents = [Span(doc, 4, 7, label='PERSON')]
    matches = [doc[start:end].text for _, start, end in matcher(doc)]
    assert len(matches) == 3
    assert matches[0] == 'Maria'
    assert matches[1] == 'Maria Esperanza'
    assert matches[2] == 'Esperanza'