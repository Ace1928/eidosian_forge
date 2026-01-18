import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.usefixtures('clean_underscore')
def test_matcher_extension_attribute(en_vocab):
    matcher = Matcher(en_vocab)
    get_is_fruit = lambda token: token.text in ('apple', 'banana')
    Token.set_extension('is_fruit', getter=get_is_fruit, force=True)
    pattern = [{'ORTH': 'an'}, {'_': {'is_fruit': True}}]
    matcher.add('HAVING_FRUIT', [pattern])
    doc = Doc(en_vocab, words=['an', 'apple'])
    matches = matcher(doc)
    assert len(matches) == 1
    doc = Doc(en_vocab, words=['an', 'aardvark'])
    matches = matcher(doc)
    assert len(matches) == 0