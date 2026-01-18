import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('pattern,text', [([{'IS_ALPHA': True}], 'a'), ([{'IS_ASCII': True}], 'a'), ([{'IS_DIGIT': True}], '1'), ([{'IS_LOWER': True}], 'a'), ([{'IS_UPPER': True}], 'A'), ([{'IS_TITLE': True}], 'Aaaa'), ([{'IS_PUNCT': True}], '.'), ([{'IS_SPACE': True}], '\n'), ([{'IS_BRACKET': True}], '['), ([{'IS_QUOTE': True}], '"'), ([{'IS_LEFT_PUNCT': True}], '``'), ([{'IS_RIGHT_PUNCT': True}], "''"), ([{'IS_STOP': True}], 'the'), ([{'SPACY': True}], 'the'), ([{'LIKE_NUM': True}], '1'), ([{'LIKE_URL': True}], 'http://example.com'), ([{'LIKE_EMAIL': True}], 'mail@example.com')])
def test_matcher_schema_token_attributes(en_vocab, pattern, text):
    matcher = Matcher(en_vocab)
    doc = Doc(en_vocab, words=text.split(' '))
    matcher.add('Rule', [pattern])
    assert len(matcher) == 1
    matches = matcher(doc)
    assert len(matches) == 1