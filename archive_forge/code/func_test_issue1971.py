import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(1971)
def test_issue1971(en_vocab):
    matcher = Matcher(en_vocab)
    pattern = [{'ORTH': 'Doe'}, {'ORTH': '!', 'OP': '?'}, {'_': {'optional': True}, 'OP': '?'}, {'ORTH': '!', 'OP': '?'}]
    Token.set_extension('optional', default=False)
    matcher.add('TEST', [pattern])
    doc = Doc(en_vocab, words=['Hello', 'John', 'Doe', '!'])
    matches = matcher(doc)
    assert all([match_id in en_vocab.strings for match_id, start, end in matches])